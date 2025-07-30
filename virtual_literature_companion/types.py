from enum import Enum


class PageType(Enum):
    """Enumeration of different page types found in books."""
    BLANK = "blank"
    TITLE_PAGE = "title_page"
    TABLE_OF_CONTENTS = "table_of_contents"
    COPYWRIGHT_PAGE = "copyright_page"
    STORY_BREAK = "story_break"
    FRONT_MATTER_BREAK = "front_matter_break"
    BACK_MATTER_BREAK = "back_matter_break"
    CONTENT = "content"


class SectionType(str, Enum):
    TABLE_OF_CONTENTS = 'table_of_contents'
    ACKNOWLEDGEMENTS = 'acknowledgements'
    BIBLIOGRAPHY = 'bibliography'
    GLOSSARY = 'glossary'
    INDEX = 'index'
    TITLE_PAGE = 'title_page'
    PREFACE = 'preface'
    EPILOGUE = 'epilogue'
    APPENDIX = 'appendix'
    DEDICATION = 'dedication'
    CHAPTER = 'chapter'
    COPYRIGHT_PAGE = 'copyright_page'