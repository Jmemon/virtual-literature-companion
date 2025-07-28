"""
Neural Page Classifier for Literary Document Processing

This module implements a modular neural network-based page classifier using BERT to automatically 
categorize different types of pages found in books and literary documents. The classifier uses 
object-oriented design patterns for better maintainability, extensibility, and testing.

Purpose:
--------
The main purpose of this module is to provide a reusable, configurable system for training 
machine learning models that can automatically classify pages from digitized books into 
predefined categories, enabling automated processing of large document collections.

Architecture:
-------------
The module is structured around several key classes:

1. **PageClassifierConfig**: Configuration dataclass that centralizes all training parameters
2. **PageDataManager**: Handles data loading, preprocessing, and dataset management
3. **MetricsComputer**: Computes comprehensive evaluation metrics with focus on non-content pages
4. **PageClassifierTrainer**: Orchestrates the entire training pipeline
5. **PageType**: Enum defining the 8 supported page categories

Key Features:
-------------
- **Modular Design**: Separated concerns with dedicated classes for data, training, and evaluation
- **Error Handling**: Comprehensive validation and error recovery throughout the pipeline
- **Logging**: Structured logging with configurable levels for debugging and monitoring
- **Configuration Management**: Centralized configuration with validation and defaults
- **Extensibility**: Easy to add new page types, models, or evaluation metrics

Supported Page Types:
--------------------
- blank: Empty or nearly empty pages
- title_page: Book title pages and covers
- table_of_contents: Table of contents and index pages
- copyright_page: Copyright and publication information
- story_break: Chapter breaks and section dividers
- front_matter_break: Preface, dedication, acknowledgments
- back_matter_break: Bibliography, appendix, author bio
- content: Main text content pages

Usage Examples:
---------------
Basic usage with default configuration:
    ```python
    from neural_page_classifier import PageClassifierTrainer
    
    trainer = PageClassifierTrainer()
    trainer.run_training_pipeline()
    ```

Custom configuration:
    ```python
    from neural_page_classifier import PageClassifierConfig, PageClassifierTrainer
    
    config = PageClassifierConfig(
        model_name="distilbert-base-uncased",
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-4
    )
    trainer = PageClassifierTrainer(config)
    trainer.run_training_pipeline()
    ```

Command line usage:
    ```bash
    python page_classifier_sidequest/neural_page_classifier.py
    ```

Input/Output:
-------------
Input: JSON files in configured raw data directory containing page objects with:
  - "text": string content of the page
  - "type": page type label matching PageType enum values

Output:
  - Trained model and tokenizer saved to configured output directory
  - Processed datasets cached for future runs
  - Comprehensive evaluation metrics and training logs
  - Performance visualizations and confusion matrices

Dependencies:
-------------
- torch: PyTorch deep learning framework
- transformers: HuggingFace transformers library  
- datasets: HuggingFace datasets library
- sklearn: Scikit-learn for metrics and data splitting
- numpy: Numerical computing
- logging: Python logging framework
- dataclasses: Configuration management
- typing: Type hints for better code documentation
"""

import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import ClassLabel, Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.nn import CrossEntropyLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PageType(Enum):
    """Enumeration of different page types found in books."""
    BLANK = "blank"
    TITLE_PAGE = "title_page"
    TABLE_OF_CONTENTS = "table_of_contents"
    COPYRIGHT_PAGE = "copyright_page"
    STORY_BREAK = "story_break"
    FRONT_MATTER_BREAK = "front_matter_break"
    BACK_MATTER_BREAK = "back_matter_break"
    CONTENT = "content"


@dataclass
class PageClassifierConfig:
    """Configuration class for the page classifier training pipeline."""
    
    # Model configuration
    model_name: str = "prajjwal1/bert-tiny"
    max_length: int = 512
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    weight_decay: float = 0.01
    non_content_weight: float = 100.0
    
    # Data splitting
    test_size: float = 0.1
    validation_size: float = 0.1  # Relative to non-test set
    rare_label_threshold: int = 3
    
    # Paths
    dataset_path: Path = field(default_factory=lambda: Path("page_classifier_sidequest/page_dataset"))
    output_dir: Path = field(default_factory=lambda: Path("page_classifier_sidequest/generated_classifiers/tinybert_page_classifier"))
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    log_level: str = "INFO"
    logging_steps: int = 10
    
    def __post_init__(self):
        """Validate configuration and set up derived paths."""
        self.raw_data_path = self.dataset_path / "raw"
        
        # Create versioned dataset directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.versioned_dataset_path = self.dataset_path / f"proc_{timestamp}"
        
        self.train_data_path = self.versioned_dataset_path / "train"
        self.val_data_path = self.versioned_dataset_path / "val"
        self.test_data_path = self.versioned_dataset_path / "test"
        
        # Validate paths
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.raw_data_path}")
        
        # Validate hyperparameters
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not 0 < self.validation_size < 1:
            raise ValueError(f"validation_size must be between 0 and 1, got {self.validation_size}")
        
        # Set random seeds
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

class PageDataManager:
    """Manages data loading, preprocessing, and dataset creation for page classification."""
    
    def __init__(self, config: PageClassifierConfig):
        self.config = config
        self.label2id = {label.value: i for i, label in enumerate(PageType)}
        self.id2label = {i: label.value for i, label in enumerate(PageType)}
    
    def load_raw_data(self) -> List[Dict]:
        """Load all JSON files from the raw data directory."""
        try:
            all_pages = []
            json_files = list(self.config.raw_data_path.glob("*.json"))
            
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {self.config.raw_data_path}")
            
            logger.info(f"Loading data from {len(json_files)} JSON files...")
            
            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_pages.extend(data)
                        else:
                            logger.warning(f"Expected list in {json_file}, got {type(data)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error reading JSON file {json_file}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error loading {json_file}: {e}")
                    continue
            
            logger.info(f"Loaded {len(all_pages)} total pages")
            return all_pages
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise
    
    def preprocess_data(self, pages: List[Dict]) -> Tuple[List[str], List[int]]:
        """Filter and preprocess page data."""
        # Filter out pages with no text
        valid_pages = []
        for page in pages:
            if not isinstance(page, dict):
                logger.warning(f"Skipping non-dict page: {type(page)}")
                continue
            
            text = page.get("text", "").strip()
            page_type = page.get("type")
            
            if not text:
                continue
                
            if page_type not in self.label2id:
                logger.warning(f"Unknown page type '{page_type}', skipping")
                continue
                
            valid_pages.append(page)
        
        logger.info(f"Filtered to {len(valid_pages)} valid pages")
        
        texts = [page["text"] for page in valid_pages]
        numeric_labels = [self.label2id[page["type"]] for page in valid_pages]
        
        return texts, numeric_labels
    
    def create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train, validation, and test datasets."""
        # Check if datasets already exist
        if self._datasets_exist():
            logger.info("Loading pre-partitioned datasets...")
            return self._load_existing_datasets()
        
        logger.info("Creating new dataset partitions...")
        
        # Load and preprocess data
        pages = self.load_raw_data()
        texts, numeric_labels = self.preprocess_data(pages)
        
        # Create balanced splits
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
            self._create_balanced_splits(texts, numeric_labels)
        )
        
        # Create HuggingFace datasets
        datasets = self._create_hf_datasets(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
        )
        
        # Save datasets
        self._save_datasets(datasets)
        
        return datasets
    
    def _datasets_exist(self) -> bool:
        """Check if pre-partitioned datasets exist, using the most recent versioned dataset."""
        # Look for existing proc_* directories
        proc_dirs = list(self.config.dataset_path.glob("proc_*"))
        if not proc_dirs:
            return False
        
        # Find the most recent directory by parsing timestamps
        def get_timestamp(proc_dir):
            try:
                timestamp_str = proc_dir.name.replace("proc_", "")
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                return datetime.min  # If parsing fails, treat as oldest
        
        most_recent = max(proc_dirs, key=get_timestamp)
        
        # Update config paths to point to most recent dataset
        self.config.versioned_dataset_path = most_recent
        self.config.train_data_path = most_recent / "train"
        self.config.val_data_path = most_recent / "val"
        self.config.test_data_path = most_recent / "test"
        
        return (
            self.config.train_data_path.exists() and 
            len(list(self.config.train_data_path.glob('*.arrow'))) > 0 and
            self.config.val_data_path.exists() and 
            len(list(self.config.val_data_path.glob('*.arrow'))) > 0 and
            self.config.test_data_path.exists() and 
            len(list(self.config.test_data_path.glob('*.arrow'))) > 0
        )
    
    def _load_existing_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load existing partitioned datasets."""
        try:
            train_dataset = Dataset.load_from_disk(str(self.config.train_data_path))
            val_dataset = Dataset.load_from_disk(str(self.config.val_data_path))
            test_dataset = Dataset.load_from_disk(str(self.config.test_data_path))
            
            # Update label mappings from dataset features
            label_feature = train_dataset.features.get('label')
            if hasattr(label_feature, 'names'):
                self.label2id = {name: i for i, name in enumerate(label_feature.names)}
                self.id2label = {i: name for i, name in enumerate(label_feature.names)}
            else:
                logger.warning("Label names not found in dataset features. Using PageType defaults.")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Failed to load existing datasets: {e}")
            raise
    
    def _create_balanced_splits(self, texts: List[str], numeric_labels: List[int]) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
        """Create balanced train/validation/test splits ensuring each split has at least one of each page type."""
        try:
            # Group samples by label
            label_groups = {}
            for i, label in enumerate(numeric_labels):
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(i)
            
            # Check which labels are present and which have insufficient samples
            all_possible_labels = set(self.id2label.keys())
            present_labels = set(label_groups.keys())
            missing_labels = all_possible_labels - present_labels
            
            if missing_labels:
                logger.warning("Some page types are missing from the dataset:")
                for label in missing_labels:
                    logger.warning(f"  {self.id2label[label]}: 0 samples")
            
            # Check for insufficient samples among present labels
            min_samples_needed = 3  # At least 1 for each split
            insufficient_labels = []
            for label, indices in label_groups.items():
                if len(indices) < min_samples_needed:
                    label_name = self.id2label[label]
                    insufficient_labels.append((label_name, len(indices)))
            
            if insufficient_labels:
                logger.warning("Insufficient samples for balanced splitting:")
                for label_name, count in insufficient_labels:
                    logger.warning(f"  {label_name}: {count} samples (need at least {min_samples_needed})")
                
                if missing_labels:
                    logger.warning("Additionally, the following page types are completely missing:")
                    for label in missing_labels:
                        logger.warning(f"  {self.id2label[label]}: 0 samples")
                
                logger.warning(f"Cannot create balanced splits. {len(insufficient_labels)} classes have insufficient samples and {len(missing_labels)} classes are missing. Continuing without them.")

            # Initialize splits
            train_indices, val_indices, test_indices = [], [], []
            splittable_labels = set()

            # For each label, ensure at least one sample in each split
            for label, indices in label_groups.items():
                # Shuffle indices for this label
                shuffled_indices = indices.copy()
                random.shuffle(shuffled_indices)

                # Allocate samples
                count = len(shuffled_indices)

                if count < min_samples_needed:
                    train_indices.extend(shuffled_indices)
                    continue

                splittable_labels.add(label)
                
                # Reserve 1 sample for test and 1 for validation
                test_indices.append(shuffled_indices[0])
                val_indices.append(shuffled_indices[1])
                
                # Remaining samples go to splits based on proportions
                remaining = shuffled_indices[2:]
                if remaining:
                    # Calculate split sizes for remaining samples
                    val_size = max(0, int(len(remaining) * self.config.validation_size) - 1)  # -1 because we already have 1
                    test_size = max(0, int(len(remaining) * self.config.test_size) - 1)      # -1 because we already have 1
                    
                    # Add remaining samples
                    if len(remaining) > val_size + test_size:
                        val_indices.extend(remaining[:val_size])
                        test_indices.extend(remaining[val_size:val_size + test_size])
                        train_indices.extend(remaining[val_size + test_size:])
                    else:
                        # If very few remaining samples, distribute evenly
                        for i, idx in enumerate(remaining):
                            if i % 3 == 0:
                                val_indices.append(idx)
                            elif i % 3 == 1:
                                test_indices.append(idx)
                            else:
                                train_indices.append(idx)
                else:
                    # If only 3 samples total, put remaining one in train
                    if count > 2:
                        train_indices.extend(shuffled_indices[2:])
            
            # Extract data using indices
            train_texts = [texts[i] for i in train_indices]
            train_labels = [numeric_labels[i] for i in train_indices]
            val_texts = [texts[i] for i in val_indices]
            val_labels = [numeric_labels[i] for i in val_indices]
            test_texts = [texts[i] for i in test_indices]
            test_labels = [numeric_labels[i] for i in test_indices]
            
            # Verify each split has all classes
            train_classes = set(train_labels)
            val_classes = set(val_labels)
            test_classes = set(test_labels)
            all_present_labels = set(numeric_labels)
            
            logger.info(f"Dataset splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
            logger.info(f"Train classes: {len(train_classes)}/{len(all_present_labels)}")
            logger.info(f"Val classes: {len(val_classes)}/{len(splittable_labels)}")
            logger.info(f"Test classes: {len(test_classes)}/{len(splittable_labels)}")
            
            if val_classes != splittable_labels or test_classes != splittable_labels:
                logger.warning("Not all classes with sufficient samples are present in val/test splits. This might indicate an issue with splitting logic.")
            
            return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
            
        except Exception as e:
            logger.error(f"Failed to create balanced splits: {e}")
            raise
    
    def _create_hf_datasets(self, train_texts: List[str], train_labels: List[int], 
                           val_texts: List[str], val_labels: List[int],
                           test_texts: List[str], test_labels: List[int]) -> Tuple[Dataset, Dataset, Dataset]:
        """Create HuggingFace Dataset objects."""
        try:
            # Create datasets
            train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
            val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
            test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
            
            # Add ClassLabel feature to preserve label names
            class_names = [pt.value for pt in PageType]
            class_label_feature = ClassLabel(names=class_names)
            
            train_dataset = train_dataset.cast_column('label', class_label_feature)
            val_dataset = val_dataset.cast_column('label', class_label_feature)
            test_dataset = test_dataset.cast_column('label', class_label_feature)
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Failed to create HuggingFace datasets: {e}")
            raise
    
    def _save_datasets(self, datasets: Tuple[Dataset, Dataset, Dataset]):
        """Save datasets to disk."""
        try:
            train_dataset, val_dataset, test_dataset = datasets
            
            logger.info("Saving partitioned datasets...")
            
            # Create directories
            self.config.train_data_path.mkdir(parents=True, exist_ok=True)
            self.config.val_data_path.mkdir(parents=True, exist_ok=True)
            self.config.test_data_path.mkdir(parents=True, exist_ok=True)
            
            # Save datasets
            train_dataset.save_to_disk(str(self.config.train_data_path))
            val_dataset.save_to_disk(str(self.config.val_data_path))
            test_dataset.save_to_disk(str(self.config.test_data_path))
            
            logger.info("Datasets saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save datasets: {e}")
            raise

class MetricsComputer:
    """Computes comprehensive evaluation metrics for page classification."""
    
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Computes detailed metrics for evaluation, including per-class accuracy,
        and precision, recall, and F1-score for non-content pages.
        """
        try:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            # Overall accuracy
            overall_accuracy = accuracy_score(labels, predictions)

            # Per-class accuracy from confusion matrix
            cm = confusion_matrix(labels, predictions, labels=list(self.id2label.keys()))
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

            # Precision, recall, F1 for non-content pages
            non_content_labels_indices = [
                idx for idx, name in self.id2label.items() if name != PageType.CONTENT.value
            ]

            # Calculate metrics for each non-content class individually
            p, r, f1, _ = precision_recall_fscore_support(
                labels,
                predictions,
                labels=non_content_labels_indices,
                average=None,
                zero_division=0,
            )

            # Assemble metrics dictionary
            metrics = {"accuracy": overall_accuracy}
            
            # Per-class accuracy
            for i, acc in enumerate(per_class_accuracy):
                if i in self.id2label:
                    metrics[f"{self.id2label[i]}_accuracy"] = acc

            # Per-class precision, recall, F1 for non-content pages
            for i, label_idx in enumerate(non_content_labels_indices):
                label_name = self.id2label[label_idx]
                metrics[f"{label_name}_precision"] = p[i]
                metrics[f"{label_name}_recall"] = r[i]
                metrics[f"{label_name}_f1"] = f1[i]

            # Macro and weighted averages for non-content classes
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                labels,
                predictions,
                labels=non_content_labels_indices,
                average="macro",
                zero_division=0,
            )
            p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
                labels,
                predictions,
                labels=non_content_labels_indices,
                average="weighted",
                zero_division=0,
            )

            metrics.update({
                "non_content_precision_macro": p_macro,
                "non_content_recall_macro": r_macro,
                "non_content_f1_macro": f1_macro,
                "non_content_precision_weighted": p_weighted,
                "non_content_recall_weighted": r_weighted,
                "non_content_f1_weighted": f1_weighted,
            })

            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {"accuracy": 0.0}


class WeightedLossTrainer(Trainer):
    """
    Custom HuggingFace Trainer that uses a weighted cross-entropy loss function.
    
    This trainer is designed to address class imbalance by applying a higher weight
    to non-content page types during loss calculation, encouraging the model to
    pay more attention to these less frequent but important classes.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the trainer, extracting custom arguments before calling the parent constructor.
        
        Args:
            non_content_weight (float): The weight to apply to all non-content classes.
            content_label_id (int): The label ID for the 'content' class.
        """
        self.non_content_weight = kwargs.pop("non_content_weight", 1.0)
        self.content_label_id = kwargs.pop("content_label_id", -1)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides the default loss computation to use a weighted cross-entropy loss.
        """
        # Pop labels from inputs to prevent the model from computing loss internally
        labels = inputs.pop("labels")
        
        # Get model outputs (logits)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        num_labels = self.model.config.num_labels
        
        # Create weights tensor, applying higher weight to non-content classes
        weights = torch.ones(num_labels, device=logits.device)
        if self.content_label_id != -1:
            for i in range(num_labels):
                if i != self.content_label_id:
                    weights[i] = self.non_content_weight
        
        # Compute loss using weighted CrossEntropyLoss
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


class PageClassifierTrainer:
    """Main trainer class that orchestrates the entire page classification pipeline."""
    
    def __init__(self, config: Optional[PageClassifierConfig] = None):
        self.config = config or PageClassifierConfig()
        self.data_manager = PageDataManager(self.config)
        self.metrics_computer = MetricsComputer(self.data_manager.id2label)
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def run_training_pipeline(self):
        """Execute the complete training pipeline."""
        try:
            logger.info("Starting page classifier training pipeline...")
            
            # Load and prepare datasets
            train_dataset, val_dataset, test_dataset = self.data_manager.create_datasets()
            
            # Initialize tokenizer and model
            self._initialize_tokenizer()
            self._initialize_model()
            
            # Tokenize datasets
            train_dataset = self._tokenize_dataset(train_dataset)
            val_dataset = self._tokenize_dataset(val_dataset)
            test_dataset = self._tokenize_dataset(test_dataset)
            
            # Initialize trainer
            self._initialize_trainer(train_dataset, val_dataset)
            
            # Train model
            logger.info("Starting model training...")
            self.trainer.train()
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            eval_results = self.trainer.evaluate(test_dataset)
            logger.info("Test set evaluation results:")
            logger.info(json.dumps(eval_results, indent=2))
            
            # Save model
            self._save_model()
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            logger.info(f"Loading tokenizer: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the model."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(self.data_manager.label2id),
                id2label=self.data_manager.id2label,
                label2id=self.data_manager.label2id
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize a dataset."""
        try:
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=self.config.max_length
                )
            
            return dataset.map(tokenize_function, batched=True)
            
        except Exception as e:
            logger.error(f"Failed to tokenize dataset: {e}")
            raise
    
    def _initialize_trainer(self, train_dataset: Dataset, val_dataset: Dataset):
        """Initialize the HuggingFace trainer."""
        try:
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="non_content_recall_macro",
                weight_decay=self.config.weight_decay,
                logging_dir=str(self.config.output_dir / 'logs'),
                logging_steps=self.config.logging_steps,
                seed=self.config.seed,
            )

            content_label_id = self.data_manager.label2id.get(PageType.CONTENT.value, -1)
            
            self.trainer = WeightedLossTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.metrics_computer.compute_metrics,
                non_content_weight=self.config.non_content_weight,
                content_label_id=content_label_id,
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise
    
    def _save_model(self):
        """Save the trained model and tokenizer."""
        try:
            logger.info(f"Saving model to {self.config.output_dir}")
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.trainer.save_model(str(self.config.output_dir))
            self.tokenizer.save_pretrained(str(self.config.output_dir))
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise


def show_data_overview():
    """Show comprehensive overview of dataset splits and class distributions."""
    try:
        config = PageClassifierConfig()
        data_manager = PageDataManager(config)
        
        # Check if datasets exist
        if not data_manager._datasets_exist():
            logger.info("Datasets not found. Loading raw data to show overview...")
            pages = data_manager.load_raw_data()
            texts, numeric_labels = data_manager.preprocess_data(pages)
            
            # Show raw data stats
            print("=== Raw Data Overview ===")
            print(f"Total pages: {len(texts)}")
            
            # Count by class
            label_counts = Counter(numeric_labels)
            print("\nClass distribution:")
            for label_id, count in sorted(label_counts.items()):
                label_name = data_manager.id2label[label_id]
                percentage = (count / len(texts)) * 100
                print(f"  {label_name:20}: {count:4d} ({percentage:5.1f}%)")
            
            print("\nNote: Run 'train' command first to create train/val/test splits")
            return
        
        # Load existing datasets
        train_dataset, val_dataset, test_dataset = data_manager._load_existing_datasets()
        
        print("=== Dataset Overview ===")
        print(f"Training set:   {len(train_dataset):4d} samples")
        print(f"Validation set: {len(val_dataset):4d} samples")
        print(f"Test set:       {len(test_dataset):4d} samples")
        print(f"Total:          {len(train_dataset) + len(val_dataset) + len(test_dataset):4d} samples")
        
        # Show class distribution for each split
        for split_name, dataset in [("Training", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
            print(f"\n=== {split_name} Set Class Distribution ===")
            
            # Count labels
            label_counts = Counter(dataset['label'])
            total = len(dataset)
            
            # Iterate over all possible labels to ensure all are shown
            for label_id in sorted(data_manager.id2label.keys()):
                count = label_counts.get(label_id, 0)
                label_name = data_manager.id2label[label_id]
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"  {label_name:20}: {count:4d} ({percentage:5.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to show data overview: {e}")
        raise


def run_training():
    """Run the training pipeline."""
    try:
        # Initialize trainer with default configuration
        trainer = PageClassifierTrainer()
        
        # Run the training pipeline
        trainer.run_training_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    """Main function for command line execution with CLI commands."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tiny_bert_page_classifier.py <command>")
        print("\nCommands:")
        print("  train  - Run the training pipeline")
        print("  data   - Show comprehensive dataset overview and class distributions")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "train":
        run_training()
    elif command == "data":
        show_data_overview()
    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands:")
        print("  train  - Run the training pipeline")
        print("  data   - Show comprehensive dataset overview and class distributions")
        sys.exit(1)


if __name__ == "__main__":
    main()
