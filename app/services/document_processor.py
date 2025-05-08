import re
import pandas as pd
from tabulate import tabulate

class DocumentProcessor:
    @staticmethod
    def extract_text(soup):
        # Define navigation-related keyword patterns
        navigation_keywords = [
            r'contact\s+us', r'click\s+(here|for)', r'guidance', r'help', r'support', r'assistance',
            r'maximize\s+screen', r'view\s+details', r'read\s+more', r'convert.*file', r'FAQ', r'learn\s+more'
        ]
        
        navigation_pattern = re.compile(r"|".join(navigation_keywords), re.IGNORECASE)

        # Remove navigation-related text
        for tag in soup.find_all("p"):
            if navigation_pattern.search(tag.text):
                tag.decompose()

        # Extract only meaningful paragraph text (excluding very short ones)
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]
        
        clean_text = "\n\n".join(paragraphs)
        
        return clean_text

    @staticmethod
    def extract_table_as_text_block(soup, file_path):
        """
        Extract tables from HTML as a single formatted text block for inclusion into page_text.
        Skips navigation tables and handles no-table cases.

        Args:
            soup (BeautifulSoup): Parsed HTML.
            file_path (str): Path to the file (for metadata).

        Returns:
            str: Formatted block of all tables from this file, or a message if no tables are found.
        """
        try:
            tables = pd.read_html(file_path)

            def is_navigation_table(table):
                """Detect if table is a 'navigation-only' table with just 'back' and 'forward'."""
                flattened = [str(cell).strip().lower() for cell in table.to_numpy().flatten()]
                navigation_keywords = {"back", "forward"}
                return set(flattened).issubset(navigation_keywords)
            
            def is_nan_only_table(table):
                """Detect if the entire table only contains NaN values."""
                return table.isna().all().all()

            table_texts = []
            table_count = 0

            for idx, table in enumerate(tables):
                if is_navigation_table(table) or is_nan_only_table(table):
                    continue
                
                if table.shape[1] == 2:
                    # Drop rows where both the second and third columns are NaN
                    table = table.dropna(how='all')

                    last_col = table.columns[-1]

                    table.loc[:, last_col] = table[last_col].fillna("")

                table_count += 1
                formatted_table = tabulate(table, headers="keys", tablefmt="grid")

                beautified_table = f"""
    ╔════════════════════════════════════════════════════╗
    ║            📊 Table {table_count} from {file_path}              ║
    ╚════════════════════════════════════════════════════╝

    {formatted_table}

    ╔════════════════════════════════════════════════════╗
    ║            🔚 End of Table {table_count}                       ║
    ╚════════════════════════════════════════════════════╝
    """
                table_texts.append(beautified_table)

            if not table_texts:
                return ""

            return "\n".join(table_texts)

        except ValueError:
            # No tables found case
            return ""

    @staticmethod
    def extract_list(soup):
        # Extract lists properly
        lists = []
        for ul in soup.find_all("ul"):
            items = [li.get_text(strip=True) for li in ul.find_all("li")]
            lists.append(items)
        return lists