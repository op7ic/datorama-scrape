from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request, urlopen
import csv

base_url = "https://www.dataroma.com"
headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}

def get_investors():

    investors_scrape = []
    req = Request(base_url , headers = headers)
    html = urlopen(req,data=None)
    soup = BeautifulSoup(html.read(),"html.parser")
    portfolio = []


    with open('datorama-holdings.csv', 'w', newline='', encoding='utf-8') as f:
	    csv_header = ['Investor','Update','Ticker','Action','Purchased Price','Current Price']
	    writer = csv.writer(f)
	    writer.writerow(csv_header)
	    for link in soup.find('span',{'id':'port_body'}).find_all('a'):
	        #print(link.get('href'))
	        update_date = str(str(link.getText()).split("Updated")[1])
	        investor = str(str(link.getText()).split("Updated")[0])

	        url_buildup = base_url+link.get('href')
	        individual_investor = Request(url_buildup,headers = headers)
	        individual_investor_html = urlopen(individual_investor)
	        soup = BeautifulSoup(individual_investor_html.read(),"html.parser")
	        
	        tbody = soup.tbody
	        tr = tbody.find_all('tr')
	        for t in tr:
	        	portfolio.append(t.text.split('\n'))
	        	line = t.text.split('\n')
	        	ticker = str(line[2]).split('-')[0]
	        	pct_porfolio = str(line[3])
	        	action = str(line[4])
	        	purchase_price = str(line[6]).strip("$")
	        	current_price = str(line[9]).strip("$")
	        	writer.writerow([investor,
	        					 update_date,
	        					 ticker,
	        					 action,
	        					 purchase_price,
	        					 current_price
	        					])
	        print("dumping:", investor)
get_investors()
