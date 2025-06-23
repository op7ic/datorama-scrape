import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime
import os
import pandas as pd

class DataromaScraper:
    def __init__(self):
        self.base_url = "https://www.dataroma.com"
        self.headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
        self.holdings_dir = "holdings"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.holdings_dir, exist_ok=True)
        os.makedirs("analysis", exist_ok=True)
    
    def scrape_dataroma(self):
        """Scrape Dataroma for all superinvestor holdings"""
        # Get list of managers
        managers_url = self.base_url + "managers.php"
        response = requests.get(managers_url,headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        all_holdings = []
        manager_links = []
        
        # Extract manager links
        for link in soup.find_all('a', href=True):
            if 'holdings.php?m=' in link['href']:
                manager_links.append(link['href'])
        
        # Scrape each manager's holdings
        for manager_link in manager_links:
            try:
                holdings = self.scrape_manager(manager_link)
                all_holdings.extend(holdings)
            except Exception as e:
                print(f"Error scraping {manager_link}: {e}")
        
        # Save holdings
        self.save_holdings(all_holdings)
        
        return all_holdings
    
    def scrape_manager(self, manager_link):
        """Scrape holdings for a specific manager"""
        manager_url = self.base_url + manager_link
        response = requests.get(manager_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holdings = []
        
        # Extract manager name and portfolio date
        manager_name = soup.find('h1').text.strip() if soup.find('h1') else "Unknown"
        
        # Extract portfolio date
        date_elem = soup.find(text=lambda text: text and 'Portfolio date:' in text)
        portfolio_date = datetime.now().strftime('%Y-%m-%d')
        if date_elem:
            try:
                date_str = date_elem.split('Portfolio date:')[1].strip()
                portfolio_date = datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
            except:
                pass
        
        # Find holdings table
        table = soup.find('table', {'id': 'grid'})
        if table:
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 7:
                    holding = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'portfolio_date': portfolio_date,
                        'manager': manager_name,
                        'stock': cols[0].text.strip(),
                        'ticker': cols[1].text.strip(),
                        'portfolio_percent': cols[2].text.strip().replace('%', ''),
                        'shares': cols[3].text.strip().replace(',', ''),
                        'recent_activity': cols[4].text.strip(),
                        'reported_price': self.clean_price(cols[5].text.strip()),
                        'current_price': self.clean_price(cols[6].text.strip())
                    }
                    holdings.append(holding)
        
        return holdings
    
    def clean_price(self, price_str):
        """Clean price string to float"""
        try:
            return float(price_str.replace('$', '').replace(',', ''))
        except:
            return None
    
    def save_holdings(self, all_holdings):
        """Save holdings with smart file management"""
        today = datetime.now().strftime('%Y%m%d')
        
        # Save daily snapshot in holdings folder
        daily_filename = os.path.join(self.holdings_dir, f'holdings_{today}.csv')
        df = pd.DataFrame(all_holdings)
        df.to_csv(daily_filename, index=False)
        
        # Save as latest in root for easy access
        df.to_csv('latest_holdings.csv', index=False)
        
        # Create a lightweight metadata file
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_holdings': len(all_holdings),
            'managers_tracked': len(df['manager'].unique()),
            'unique_stocks': len(df['ticker'].unique()),
            'file_location': daily_filename
        }
        
        with open('scrape_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(all_holdings)} holdings to {daily_filename}")

def main():
    scraper = DataromaScraper()
    scraper.scrape_dataroma()

if __name__ == "__main__":
    main()