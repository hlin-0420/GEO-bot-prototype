def auto_adjust_column_width(writer, df):
    """ Auto-adjusts column width based on the max length of cell content in each column """
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    for column in df.columns:
        max_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
        col_idx = df.columns.get_loc(column) + 1
        worksheet.column_dimensions[chr(64 + col_idx)].width = max_length