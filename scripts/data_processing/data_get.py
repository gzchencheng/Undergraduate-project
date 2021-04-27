#通过open-i提供的api抓取多种影像报告
import requests
import urllib.request
import os
import time

requests.adapters.DEFAULT_RETRIES = 5

url_head = "https://openi.nlm.nih.gov/api/search?"
#由oepn-i提供的request url头部
data_file = "data/"
img_types = ['c','x','m']
#抓取CT,X-Ray,MRI三种影像学报告

if not os.listdir(data_file):
    for img_type in img_types:
        os.mkdir(data_file+img_type)
        os.mkdir(data_file + img_type + '/image')
        os.mkdir(data_file + img_type + '/report')

for img_type in img_types:
    img_number = 0
    img_file = data_file+img_type+'/image/'
    report_file = data_file+img_type+'/report/'
    for i in range(1,100):
        url = url_head + "m={m}&n={n}&it={it}".format(m=100*(i-1)+1,n=i*100,it=img_type)
        r = requests.get(url)
        response = r.json()
        report_list = response['list']
        r.close()
        for report in report_list:
            s = requests.session()
            s.keep_alive = False
            if report["docSource"] != 'MPX':
            #仅收集由MedPix提供的规范数据
                continue
            img_number += 1
            time.sleep(3)
            rr = requests.get("https://openi.nlm.nih.gov" + report['detailedQueryURL'])
            detail = rr.json()['list'][0]
            abstract = detail["abstract"]
            img = "https://openi.nlm.nih.gov"+detail["imgLarge"]
            f = open(report_file+'{n}.html'.format(n=str(img_number)),'w',encoding='utf-8')
            f.write(abstract)
            urllib.request.urlretrieve(img,img_file+"{n}.png".format(n=str(img_number)))
            rr.close()