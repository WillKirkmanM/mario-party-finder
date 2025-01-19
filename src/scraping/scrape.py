import requests
from bs4 import BeautifulSoup
import os
import time

BASE_URL = 'https://www.mariowiki.com'

GAME_URLS = {
    'Mario Party': '/List_of_Mario_Party_minigames',
    'Mario Party 2': '/List_of_Mario_Party_2_minigames', 
    'Mario Party 3': '/List_of_Mario_Party_3_minigames',
    'Mario Party 4': '/List_of_Mario_Party_4_minigames',
    'Mario Party 5': '/List_of_Mario_Party_5_minigames',
    'Mario Party 6': '/List_of_Mario_Party_6_minigames',
    'Mario Party 7': '/List_of_Mario_Party_7_minigames',
    'Mario Party 8': '/List_of_Mario_Party_8_minigames',
    'Mario Party 9': '/List_of_Mario_Party_9_minigames',
    'Mario Party 10': '/List_of_Mario_Party_10_minigames',
    'Mario Party DS': '/List_of_Mario_Party_DS_minigames',
    'Mario Party Advance': '/List_of_Mario_Party_Advance_minigames',
    'Mario Party-e': '/List_of_Mario_Party-e_minigames',
    'Mario Party: Island Tour': '/List_of_Mario_Party:_Island_Tour_minigames',
    'Mario Party: Star Rush': '/List_of_Mario_Party:_Star_Rush_minigames', 
    'Mario Party: The Top 100': '/List_of_Mario_Party:_The_Top_100_minigames',
    'Super Mario Party': '/List_of_Super_Mario_Party_minigames',
    'Mario Party Superstars': '/List_of_Mario_Party_Superstars_minigames'
}

def download_image_with_retry(url, max_attempts=3, delay=1):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers)
            print(f"Trying URL: {url}")
            print(f"Status Code: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type', '')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    return response.content
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(delay)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay)
    return None

def clean_image_url(url):
    """Convert thumbnail URL to full resolution URL"""
    EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
    
    url_lower = url.lower()
    
    for ext in EXTENSIONS:
        if url_lower.count(ext) > 1:
            first_ext_pos = url_lower.find(ext)
            url = url[:first_ext_pos + len(ext)]
            break
            
    url = url.replace('/thumb/', '/')
    
    parts = url.split('/')
    if parts[-1].startswith(tuple(f"{i}px-" for i in range(1000))):
        parts[-1] = parts[-1].split('-', 1)[1]
    
    return '/'.join(parts)

def get_unique_filename(base_path, name):
    """Generate unique filename with incrementing number if file exists"""
    filename = f"{name}.webp"
    filepath = os.path.join(base_path, filename)
    
    counter = 1
    while os.path.exists(filepath):
        filename = f"{name}_{counter}.webp"
        filepath = os.path.join(base_path, filename)
        counter += 1
        
    return filepath

def scrape_mariowiki():
    """Original mariowiki scraping logic"""
    os.makedirs('data/train', exist_ok=True)
    
    for game_name, minigames_path in GAME_URLS.items():
        url = BASE_URL + minigames_path
        print(f"Scraping {game_name} from {url}")
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        gallery_items = soup.find_all('div', class_='thumb')
        
        for item in gallery_items:
            try:
                img_element = item.find('img')
                img_url = img_element['src']
                if not img_url.startswith('http'):
                    img_url = 'https:' + img_url
                
                img_url = clean_image_url(img_url)
                
                name_element = item.find_next('div', class_='gallerytext').find('a')
                minigame_name = name_element.get_text()
                
                game_dir = os.path.join('data/train', game_name)
                os.makedirs(game_dir, exist_ok=True)
                
                filepath = get_unique_filename(game_dir, minigame_name)
                
                img_data = download_image_with_retry(img_url)
                if img_data:
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                    print(f"Successfully downloaded: {game_name}/{filepath}")
                else:
                    print(f"Failed to download after retries: {game_name}/{filepath}")
                    
            except Exception as e:
                print(f"Error processing minigame: {e}")

def scrape_mariopartylegacy():
    """Scrape from mariopartylegacy.com"""
    os.makedirs('data/train', exist_ok=True)
    
    BASE_LEGACY_URL = 'https://mariopartylegacy.com'
    GAMES = [
        'mario-party-1', 'mario-party-2', 'mario-party-3', 'mario-party-4',
        'mario-party-5', 'mario-party-6', 'mario-party-7', 'mario-party-8',
        'mario-party-9', 'mario-party-10', 'mario-party-advance', 
        'mario-party-ds', 'mario-party-island-tour', 'mario-party-star-rush',
        'mario-party-the-top-100', 'super-mario-party', 'mario-party-superstars',
        'super-mario-party-jamboree'
    ]
    
    for game in GAMES:
        url = f"{BASE_LEGACY_URL}/{game}/minigame-list-tips-and-unlockables/"
        print(f"Scraping {game} from {url}")
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            rows = soup.find_all('tr')
            for row in rows:
                try:
                    img_element = row.find('img')
                    if not img_element:
                        continue
                        
                    img_url = img_element.parent.get('href')
                    name_element = row.find('strong')
                    if not name_element:
                        continue
                        
                    minigame_name = name_element.text.strip()
                    
                    game_dir = os.path.join('data/train', game.replace('-', ' ').title())
                    os.makedirs(game_dir, exist_ok=True)
                    
                    filepath = get_unique_filename(game_dir, minigame_name)
                    
                    img_data = download_image_with_retry(img_url)
                    if img_data:
                        with open(filepath, 'wb') as f:
                            f.write(img_data)
                        print(f"Successfully downloaded: {game}/{filepath}")
                    else:
                        print(f"Failed to download after retries: {game}/{filepath}")
                        
                except Exception as e:
                    print(f"Error processing minigame row: {e}")
                    
        except Exception as e:
            print(f"Error processing game {game}: {e}")

def scrape_minigames():
    """Main function to run both scrapers"""
    scrape_mariowiki()
    scrape_mariopartylegacy()

if __name__ == '__main__':
    scrape_minigames()