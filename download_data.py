from selenium import webdriver

from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC

import time
from utilities.loaders import download_dataset

from concurrent.futures import ThreadPoolExecutor


if __name__ == "__main__":
    chrome_options = ChromeOptions()
    # service = ChromeService(executable_path="C:/Executables/chromedriver-win64/chromedriver.exe")
    # chrome_options.add_experimental_option('detach', True)    
    service = ChromeService(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get('http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/')

    
    time.sleep(5)

    
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)") 

    
    anchor_tags = driver.find_elements(By.TAG_NAME, "a")
    anchor_tags

    def helper(a_tag):
        # this will extract the href of all acnhor tags 
        link = a_tag.get_attribute('href')
        return link

    # concurrently read and load all .json files
    with ThreadPoolExecutor() as exe:
        links = list(exe.map(helper, anchor_tags))

    # exclude all hrefs without .tgz extension
    download_links = list(filter(lambda link: link.endswith('.tgz'), links))

    download_dataset(download_links)