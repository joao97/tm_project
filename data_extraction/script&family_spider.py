#Authors:
# Filipa Sá da Costa
# Guilherme Martins
# Liah Rosenfeld (20180044)
# João Fernandes (20180061)

import scrapy
import re 
from scrapy.crawler import CrawlerProcess
import numpy as np 

#Creating an empty dictionary that will contain the scripts of all the episodes
episode_script = {}
words = []
#This code extracts the scripts as well as family words
class script_spider(scrapy.Spider):
    name = "scripts"

    def start_requests(self):
        for i in range(1,8):
            yield scrapy.Request(url='https://genius.com/albums/Game-of-thrones/Season-'+str(i)+'-scripts', callback=self.parse, meta={'season':i})
        yield scrapy.Request(url='http://learnersdictionary.com/3000-words/topic/family-members/1',callback = self.parse_family_words)
        yield scrapy.Request(url='http://learnersdictionary.com/3000-words/topic/family-members/2',callback = self.parse_family_words)
    def parse(self, response):
        episodes = response.xpath('//div[@class= "chart_row chart_row--light_border chart_row--full_bleed_left chart_row--align_baseline chart_row--no_hover"]')
        #Number
        for episode in episodes:
            ep = episode.xpath('*/span[@class = "chart_row-number_container-number chart_row-number_container-number--gray"]/span/text()').get()
            url = episode.xpath('*/a/@href').get()
            yield scrapy.Request(url, callback=self.parse_episode, meta = {'episode': ep , 'season': response.meta.get('season')})
        
    def parse_episode(self, response):        
        global episode_script
        episode_script['s'+str(response.meta.get('season'))+'ep'+str(response.meta.get('episode'))]=response.xpath('//div[@class = "lyrics"]').extract()
        
    def parse_family_words(self,response):
        global words
        words = words + response.xpath("//ul[@class = 't_words']/li/a/text()").getall()
#Running spider        
process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})
process.crawl(script_spider)
process.start(True)

#saving the dictionary to a numpy file
np.save('scripts_processing/scripts.npy',episode_script)
words = [re.findall('\w+-?\w+',word)[0] for word in words if re.search('\w+-?\w+',word)]
np.save('scripts_processing/family_words.npy',words)