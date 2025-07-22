# Design Document

## Overview

This design document outlines the approach to fix the syntax error in the app.py file. The error is caused by CSS styling code that appears to be duplicated and improperly placed in the Python file, causing a SyntaxError with an "invalid decimal literal" message on line 1235.

## Architecture

The application uses Gradio for the UI components and FastAPI for the backend. The CSS styling is embedded directly in the Python code using HTML strings passed to Gradio components. The fix will maintain this architecture while ensuring the CSS is properly contained within string literals.

## Components and Interfaces

### CSS Styling Component

The CSS styling is embedded in the application through a `gr.HTML()` component. The issue is that there appears to be a large block of CSS that is duplicated - once inside the `gr.HTML()` call and then again directly in the Python code outside of any function or string literal.

The fix will:
1. Remove the duplicated CSS that's outside of any string literal
2. Ensure all CSS is properly contained within the `gr.HTML()` component
3. Maintain all the original styling rules and classes

## Error Analysis

The specific error "SyntaxError: invalid decimal literal" on line 1235 indicates that Python is interpreting part of the CSS code as Python code. This happens because the CSS code is not properly contained within a string literal or function call.

Line 1235 contains:
```
--shadow-soft: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
```

Python is interpreting `0.05` as a decimal literal, but the preceding text makes it invalid Python syntax.

## Implementation Strategy

1. Identify the boundaries of the duplicated CSS code
2. Remove the duplicated CSS code that's outside of string literals
3. Ensure the CSS within the `gr.HTML()` component is complete and properly formatted
4. Verify that all styling rules are preserved

## Testing Strategy

1. Syntax validation: Ensure the Python file can be parsed without syntax errors
2. Visual inspection: Run the application and verify that the styling is applied correctly
3. Functionality testing: Verify that all UI components work as expected

## Error Prevention

To prevent similar issues in the future:
1. Consider moving large CSS blocks to separate files
2. Use clear comments to mark the beginning and end of embedded CSS
3. Consider using a CSS preprocessor or a more structured approach to styling