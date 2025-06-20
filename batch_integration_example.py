#!/usr/bin/env python3
"""
Example showing how to integrate SuryaTableWrapper with batch processing
This demonstrates how to modify the batch_analyze.py code to use dynamic language
"""

# Example modification for batch_analyze.py
def table_processing_example():
    """
    This shows how the table processing section in batch_analyze.py 
    could be modified to use the SuryaTableWrapper with dynamic language
    """
    
    # Initialize the table model (this would be done once)
    from magic_pdf.model.sub_modules.table.surya_table.surya_table_wrapper import SuryaTableWrapper
    
    # Initialize with default configuration
    config = {
        'disable_tqdm': True,
        'language': 'en',  # Default language
        'detection_batch_size': 4,
        'recognition_batch_size': 32,
        'table_rec_batch_size': 6
    }
    table_model = SuryaTableWrapper(config=config)
    
    # Simulate the batch processing loop
    table_res_list_all_page = [
        {'lang': 'en', 'table_img': 'english_table.png'},
        {'lang': 'zh', 'table_img': 'chinese_table.png'},
        {'lang': 'ja', 'table_img': 'japanese_table.png'},
    ]
    
    # Process each table with its specific language
    for table_res_dict in table_res_list_all_page:
        _lang = table_res_dict['lang']  # Language from batch processing
        table_img = table_res_dict['table_img']
        
        # Use the dynamic language parameter
        html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(
            table_img, 
            language=_lang  # Pass the language dynamically
        )
        
        # Process the results
        if html_code:
            expected_ending = html_code.strip().endswith('</html>') or html_code.strip().endswith('</table>')
            if expected_ending:
                table_res_dict['table_res']['html'] = html_code
                print(f"✓ Successfully processed {_lang} table in {elapse:.2f}s")
            else:
                print(f"✗ Table processing failed for {_lang} table")
        else:
            print(f"✗ No table detected for {_lang} table")


# Alternative approach: Language-specific model instances
def language_specific_models_example():
    """
    Alternative approach using separate model instances for different languages
    """
    
    from magic_pdf.model.sub_modules.table.surya_table.surya_table_wrapper import SuryaTableWrapper
    
    # Create language-specific models
    models = {}
    languages = ['en', 'zh', 'ja', 'ko']
    
    for lang in languages:
        config = {
            'disable_tqdm': True,
            'language': lang,
            'detection_batch_size': 4,
            'recognition_batch_size': 32,
            'table_rec_batch_size': 6
        }
        models[lang] = SuryaTableWrapper(config=config)
    
    # Process tables using language-specific models
    table_res_list_all_page = [
        {'lang': 'en', 'table_img': 'english_table.png'},
        {'lang': 'zh', 'table_img': 'chinese_table.png'},
        {'lang': 'ja', 'table_img': 'japanese_table.png'},
    ]
    
    for table_res_dict in table_res_list_all_page:
        _lang = table_res_dict['lang']
        table_img = table_res_dict['table_img']
        
        # Use the appropriate model for this language
        if _lang in models:
            model = models[_lang]
            html_code, table_cell_bboxes, logic_points, elapse = model.predict(table_img)
            
            if html_code:
                print(f"✓ Processed {_lang} table using language-specific model")
        else:
            print(f"✗ No model available for language: {_lang}")


if __name__ == "__main__":
    print("SuryaTableWrapper Batch Integration Examples")
    print("=" * 50)
    
    print("\n1. Dynamic Language Parameter Approach:")
    print("   - Single model instance")
    print("   - Language passed per prediction")
    print("   - More memory efficient")
    
    print("\n2. Language-Specific Models Approach:")
    print("   - Separate model instances per language")
    print("   - Potentially better performance")
    print("   - Higher memory usage")
    
    print("\nBoth approaches are compatible with the batch_analyze.py workflow.")
    print("Choose based on your memory and performance requirements.") 