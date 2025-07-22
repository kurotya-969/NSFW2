# Requirements Document

## Introduction

This feature aims to fix issues with Japanese output in the app.py file. Currently, there are problems with meta comments appearing in the output and issues with the Japanese text formatting. The goal is to improve the clean_meta function to properly handle Japanese text and remove all meta comments from the output.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to improve the clean_meta function to properly handle Japanese text and remove all meta comments, so that the output is clean and natural.

#### Acceptance Criteria

1. WHEN the LLM outputs Japanese text with meta comments THEN the system SHALL remove all meta comments completely
2. WHEN the LLM outputs Japanese text with parenthetical expressions (both Japanese and English) THEN the system SHALL remove them completely
3. WHEN the LLM outputs Japanese text with line prefixes like "Note:", "補足:", etc. THEN the system SHALL remove them completely
4. WHEN the LLM outputs Japanese text with explanatory phrases THEN the system SHALL remove them completely
5. WHEN the LLM outputs Japanese text with multiple consecutive spaces or Japanese full-width spaces THEN the system SHALL normalize them properly
6. WHEN the clean_meta function processes text THEN it SHALL preserve the natural flow of Japanese conversation

### Requirement 2

**User Story:** As a developer, I want to enhance the regex patterns in the clean_meta function to catch more meta comment patterns, so that no meta comments appear in the output.

#### Acceptance Criteria

1. WHEN the LLM outputs text with new meta comment patterns THEN the system SHALL identify and remove them
2. WHEN the LLM outputs text with Japanese-specific meta comment patterns THEN the system SHALL identify and remove them
3. WHEN the LLM outputs text with nested or complex meta comment patterns THEN the system SHALL identify and remove them
4. WHEN the clean_meta function is updated THEN it SHALL maintain backward compatibility with existing patterns

### Requirement 3

**User Story:** As a developer, I want to improve the system prompt to explicitly instruct the LLM not to output meta comments, so that the need for cleaning is reduced.

#### Acceptance Criteria

1. WHEN the system prompt is updated THEN it SHALL include clear instructions to avoid meta comments
2. WHEN the system prompt is updated THEN it SHALL include specific examples of prohibited meta comment patterns
3. WHEN the system prompt is updated THEN it SHALL maintain the character's personality and behavior
4. WHEN the system prompt is updated THEN it SHALL emphasize the importance of natural Japanese conversation without meta comments