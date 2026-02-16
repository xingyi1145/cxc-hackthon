import time
import csv
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from selenium.webdriver.common.by import By

cur_id = 0

def get_page_source(driver, page_number):
    # URL to scrape
    url = f"https://www.remax.ca/find-real-estate?lang=en&pageNumber={page_number}"

    # Open the URL
    driver.get(url)

    # Wait for the page to load
    time.sleep(10)

    # Get the page source
    page_source = driver.page_source

    return page_source

def parse_listings(page_source):
    global cur_id
    soup = BeautifulSoup(page_source, 'html.parser')
    listings = soup.find_all('div', {'data-testid': 'listing-card'})
    geolocator = Nominatim(user_agent="cxc_scraper")

    extracted_data = []
    for listing in listings:
        try:
            price_element = listing.find('h2', {'class': 'listing-card_price__lEBmo'})
            price = price_element.text.strip() if price_element else 'N/A'

            address_element = listing.find('div', {'data-cy': 'property-address'})
            address = address_element.text.strip() if address_element else 'N/A'
            
            city = address.split(',')[-2].strip() if ',' in address else 'N/A'

            latitude = 'N/A'
            longitude = 'N/A'
            if address != 'N/A':
                try:
                    location = geolocator.geocode(address)
                    if location:
                        latitude = location.latitude
                        longitude = location.longitude
                    time.sleep(1)
                except Exception as e:
                    print(f"Error geocoding {address}: {e}")

            bedrooms_element = listing.find('span', {'data-cy': 'property-beds'})
            bedrooms = bedrooms_element.find('span').text if bedrooms_element else 'N/A'

            bathrooms_element = listing.find('span', {'data-cy': 'property-baths'})
            bathrooms = bathrooms_element.find('span').text if bathrooms_element else 'N/A'

            if (latitude != "N/A"):
                data = {
                    'id': cur_id,
                    'city': city,
                    'address': address,
                    'longitude': longitude,
                    'latitude': latitude,
                    'price': price,
                    'bedrooms': bedrooms,
                    'bathoom': bathrooms,
                }
                extracted_data.append(data)
                cur_id+=1

        except Exception as e:
            print(f"Error parsing listing: {e}")
            continue

    return extracted_data

def write_to_json(data):
    if not data:
        return

    with open('listings.json', 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Set up the Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    all_listings_data = []
    for page_number in range(1, 101):
        try:
            print(f"Scraping page {page_number}...")
            page_source = get_page_source(driver, page_number)
            listings_data = parse_listings(page_source)
            if not listings_data:
                print(f"No listings found on page {page_number}. Stopping.")
                break
            all_listings_data.extend(listings_data)
        except Exception as e:
            print(f"Error scraping page {page_number}: {e}")
            continue

    # Close the browser
    driver.quit()

    write_to_json(all_listings_data)
    print("Scraping complete. Data saved to listings.json")