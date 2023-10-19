import json
import math
import os
import shutil
import sys
import time
import undetected_chromedriver as uc

import requests
import unidecode
from bs4 import BeautifulSoup
from bs4.element import Tag
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from Lib import datetime

if sys.version_info[0] > 2:
    pass
else:
    reload(sys)
    sys.setdefaultencoding('utf8')

'''
Commandline based Google Images scraping/downloading. Gets up to 1000 images.
Author: Rushil Srivastava (rushu0922@gmail.com)
'''


def search(url, header):
    # Create a browser and resize depending on user preference

    if header:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
    else:
        chrome_options = None

    browser = webdriver.Chrome(options=chrome_options)
    browser.set_window_size(1024, 768)
    print("\n===============================================\n")
    print("[%] Successfully launched ChromeDriver")

    # Open the link
    browser.get(url)
    time.sleep(1)
    print("[%] Successfully opened link.")

    element = browser.find_element(By.TAG_NAME,'body')

    print("[%] Scrolling down.")
    # Scroll down
    for i in range(30):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)  # bot id protection

    try:
        browser.find_element(browser,By.ID,'smb').click()
        print("[%] Successfully clicked 'Show More Button'.")
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection

    print("[%] Reached end of Page.")

    time.sleep(1)
    # Get page source and close the browser
    source = browser.page_source
    # Removed saving
    # if sys.version_info[0] > 2:
    #     with open('dataset/logs/google/source.html', 'w+', encoding='utf-8', errors='replace') as f:
    #         f.write(source)
    # else:
    #     with io.open('dataset/logs/google/source.html', 'w+', encoding='utf-8') as f:
    #         f.write(source)

    browser.close()
    print("[%] Closed ChromeDriver.")

    return source


def error(link):
    print("[!] Skipping {}. Can't download or no metadata.\n".format(link))
    # file = Path("dataset/logs/google/errors.log".format(query))
    # if file.is_file():
    #     with open("dataset/logs/google/errors.log".format(query), "a") as myfile:
    #         myfile.write(link + "\n")
    # else:
    #     with open("dataset/logs/google/errors.log".format(query), "w+") as myfile:
    # myfile.write(link + "\n")


def save_image(link, file_path):
    ua = UserAgent()
    headers = {"User-Agent": ua.random}
    r = requests.get(link, stream=True, headers=headers)
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise Exception("Image returned a {} error.".format(r.status_code))

def merge(path,path2,path3):
    from PIL import Image
#Read the two images
    image1 = Image.open(path)
    image2 = Image.open(path2)
#resize, first image
    x=0
    y=0
    if(image1.size[0]>image2.size[0]):
        x=image1.size[0]
    else:
        x=image2.size[0]
    if(image1.size[1]>image2.size[1]):
        y=image1.size[1]
    else:
        y=image2.size[1]
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB',((image1_size[0]+image2_size[0]),y), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save(path3,"JPEG")

def download_image(link, path):

    # Use a random user agent header for bot id
    ua = UserAgent()
    headers = {"User-Agent": ua.random}

    # Get the image link
    try:
        # Get the file name and type
        file_name = link.split("/")[-1]
        type = file_name.split(".")[-1]
        type = (type[:3]) if len(type) > 3 else type
        if type.lower() == "jpe":
            type = "jpeg"
        if type.lower() not in ["jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
            type = "jpg"

        # Download the image
        print("[%] Downloading Image #{} from {}".format(
            "", link))
        try:
            save_image(link, path + "/{}.{}".format("", type), headers)
            print("[%] Downloaded File")
            # if metadata:
            #     with open("dataset/google/{}/Scrapper_{}.json".format(query, str(download_image.delta)), "w") as outfile:
            #         json.dump(image_data, outfile, indent=4)
        except Exception as e:
            print("[!] Issue Downloading: {}\n[!] Error: {}".format(link, e))
            error(link)
    except Exception as e:
        print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))
        error(link)


    # Open the link




def teams():
   # chrome_options = Options()
    #browser = webdriver.Chrome(options=chrome_options)
    options = uc.ChromeOptions()
    options.headless = False  # Set headless to False to run in non-headless mode
    listt=[]

    with open('C:/Users/Dylan/Desktop/modelsF.json') as f:
       listt = json.load(f)
    #print(len(listt))
    #print(listt)

    browser = uc.Chrome(use_subprocess=True, options=options)
    browser.set_window_size(2300,1000)
    browser.get("https://www.google.com/search?q=teamskeet&sourceid=chrome&ie=UTF-8")
    time.sleep(60)
    url=browser.current_url
    url2=str(url)+"&page="
    time.sleep(1)
    source = browser.page_source
    soup = BeautifulSoup(str(source), "html.parser")

    result=float(str(soup.find("span",class_="results-bar__count-digit mpjs-result-count").get_text()))
    result2=float(result/60)
    result2=int(math.ceil(result2))
    starti=int(math.floor(float(len(listt))/60))
    remainder=len(listt)-starti*60
    i=starti
    index=0
    for starti in range(result2):
        i=i+1
        browser.get(str(url2)+str(i))
        time.sleep(1)
        source = browser.page_source
        soup = BeautifulSoup(str(source), "html.parser")
        for a in soup.find_all("div", class_="card card--model"):
            if(remainder>0):
                remainder=remainder-1
                continue
            model=Model(str(a))
            model.gender="F"
            model.url=str(a.find("a",class_="card__thumbnail__link")['href']).lstrip().rstrip()
            #print(str(a.find("a",class_="card__stats__title ")))
            #print(str(a.find("span",class_="card__id")))
            #print(str(a.findNext("span",class_="card__stats__count")))
            model.rank=str(a.find("span",class_="card__id").get_text()).lstrip().rstrip().replace("#","")
            model.movies=str(a.findNext("span",class_="card__stats__count").get_text()).lstrip().rstrip()
            browser.get(model.url)
            time.sleep(1)
            source = browser.page_source
            soup = BeautifulSoup(str(source), "html.parser")

            model.name=str(soup.find("h1",class_="profile__name").get_text()).lstrip().rstrip().replace("\"","").replace(".","").replace("ñ","n").replace("ñ","n")
            print(str(model.rank)+"-"+str(model.name))
            model.downloadimages()

            if(model.gender=="F"):
             browser.get(str(model.phuburl()))
             time.sleep(1)
             source2 = browser.page_source
             soup2=BeautifulSoup(str(source2), "html.parser")
             try:
                 p3="C:/Users/Dylan/Desktop/modelpics/"+model.name+"-3.jpg"
                 p4="C:/Users/Dylan/Desktop/modelpics/"+model.name+"-4.jpg"
                 save_image(str(soup2.find("img",class_="jcrop-preview")['src']),p3)
                 save_image(str(soup2.find(id="coverPictureDefault")['src']),p4)

             except:
                #print("PHNOTFOUND")
                source2=""

             if(source2!=""):
                 model.phrank=  str(searchany(str(source2),"div","infoBoxes","infoBox rankDetails")).split("\">")[1].split("<")[0]
                 try:
                  model.phmovies=str(soup2.find_all("div",class_="showingInfo")[1].get_text()).lstrip().rstrip().split("of ")[1]
                 except:
                    try:
                     model.phmovies=str(soup2.find_all("div",class_="showingInfo")[0].get_text()).lstrip().rstrip().split("of ")[1]
                    except:
                     model.phmovies="000"
                 model.bio=str(searchany(str(source2),"div","biographyText column text js-bioText","description"))
                 model.about=str(searchany(str(source2),"div","biographyAbout column text js-bioAbout","<div>"))
                 try:
                  model.eye=str(searchany2(str(source2),"div","infoPiece","Eye Color:"))
                 except:
                     #print("ERROR-EYE2")
                     index=index

             try:
                x=str(searchlist(str(source),"Measurements")).split("-")

                model.hips=x[2]
                model.waist=x[1]
                model.breastsize=x[0]
             except:
                #print("ERROR-MEASUREMENTS")
                if(source2!=""):
                  try:
                    x=str(searchany2(str(source2),"div","infoPiece","Measurements:")).split("-")
                    model.hips=x[2]
                    model.waist=x[1]
                    model.breastsize=x[0]
                  except:
                      #print("ERROR-MEASUREMENTS2")
                      index=index


             model.hair=str(searchlist(str(source),"Hair Color"))
             if (model.hair == "") & (source2 != ""):
                 model.hair=str(searchany2(str(source2),"div","infoPiece","Hair Color:"))

             model.piercing=str(searchlist(str(source),"Piercings"))
             if(model.piercing=="")& (source2!=""):
                 model.piercing=str(searchany2(str(source2),"div","infoPiece","Piercings:"))




             model.ethnicity=str(searchlist(str(source),"Ethnicity"))
             if(model.ethnicity=="")& (source2!=""):
                model.ethnicity=str(searchany2(str(source2),"div","infoPiece","Ethnicity:"))

             model.nationality=str(searchlist(str(source),"Nationality"))
             if(model.nationality=="")& (source2!=""):
                model.nationality=str(searchany2(str(source2),"div","infoPiece","Background:"))

             try:
                model.dob=Date(str(searchlist(str(source),"Gender, Date of birth")).split(", ")[1]).dict()

             except:
                #print("ERROR-DOB")
                index=index

             try:
                x=str(searchlist(str(source),"Size")).split(", ")
                model.weight=x[1].replace(" lbs.","")
                feet=int(x[0].split("'")[0])*12+int(x[0].split("'")[1].split("\"")[0])
                model.height=str(feet)
             except:
                #print("ERROR-SIZE")
                if(source2!=""):
                    try:
                        model.height=str(searchany2(str(source2),"div","infoPiece","Height:"))
                        feet=int(model.height.split(" ft")[0])*12+int(model.height.split("ft ")[1].split(" in")[0])
                        model.height=str(feet)
                    except:
                        #print("ERROR-HEIGHT2")
                        index=index

                    try:
                        model.weight=str(searchany2(str(source2),"div","infoPiece","Weight:"))
                        model.weight=model.weight.split(" lbs")[0]
                    except:
                        #print("ERROR-WEIGHT2")
                        index=index

             try:
                x=str(searchlist(str(source),"Breasts")).split(", ")
                model.breasttype=x[0]
                if(str(model.breastsize)==""):
                    model.breastsize=x[1]
             except:
                #print("ERROR-BREASTS")
                index=index


             if(source2!=""):
                 try:
                     fake=str(searchany2(str(source2),"div","infoPiece","Fake Boobs:"))
                     if(fake=="Yes"):
                         model.breasttype="Fake"
                     elif(fake=="No"):
                         model.breasttype="Natural"
                 except:
                     #print("ERROR-BOOOB2")
                     index=index


             model.tattoo=str(searchlist(str(source),"Tattoos"))
             if(model.tattoo=="")& (source2!=""):
                 model.tattoo=str(searchany2(str(source2),"div","infoPiece","Tattoos:"))

            listt.append(model.dictcustom())
            with open("C:\\Users\\Dylan\\Desktop\\models"+str(model.gender)+".json", "w") as write_file:
                json.dump(listt,write_file, indent=4)

"""
for b in soup.find_all("a", class_=""):
link= str(b).split("href=\"")[1]
link2=link.split("\"")[0]

for i in range(21650):
if i>7350:
try:
 url="https://members.teamskeet.com/p/"+i.__str__()
 browser.get(url)
 time.sleep(1)
 source = browser.page_source
 soup = BeautifulSoup(str(source), "html.parser")
 for a in soup.find_all("div", class_="movie-bar__group movie-bar__options"):
     for b in a.find_all("a", class_=""):
         link= str(b).split("href=\"")[1]
         link2=link.split("\" target")[0]
         link3=link2.split(".zip")[0]
         link4=link3.split("/")
         link5=link4[len(link4)-1]
         browser.get(link2.replace("amp;",""))
         time.sleep(1)
         data= (link5,str(soup))
         print(i.__str__()+"-"+link5)
         with open("C:\\Users\\Dylan\\Desktop\\tsjson\\"+i.__str__()+"-"+link5+".json", "w") as write_file:
             json.dump(data, write_file)
except:
 data= (i,str(soup))
 print(i.__str__())
 with open("C:\\Users\\Dylan\\Desktop\\tsjson\\"+i.__str__()+".json", "w") as write_file:
     json.dump(data, write_file)
     
     """

def searchany2(source,x,y,search):
    p=""
    try:
        soup = BeautifulSoup(str(source), "html.parser")
        header_elements = soup.find_all("div",class_="infoPiece")
        for header_element in header_elements:
         if(str(header_element).__contains__(str(search))):
          p=str(header_element).split("\">")[2].split("<")[0]
        #print("return-"+str(p))
        return str(p)
    except:
        #print("ERROR-CANNOT FIND IN LIST-"+search)
        return ""


def cropblack(path):
    import cv2
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    cv2.imwrite(str(path),crop)

def gif():
    from PIL import Image

   # merge("C:/Users/Dylan/Downloads/elsa_jean.jpg","C:/Users/Dylan/Downloads/elsa_jean (1).jpg")

    imageObject = Image.open("C:/Users/Dylan/Pictures/daddys-lap/daddys-lap-309.jpg")
    #print(imageObject.is_animated)

    print(imageObject.n_frames)



# Display individual frames from the loaded animated GIF file

    for frame in range(0,imageObject.n_frames):

        imageObject.seek(frame)
        #imageObject.save("C:/Users/Dylan/Desktop/cp/"+str(frame)+".jpg")

       # imageObject.show()

class Date:

    def __init__(self,fulldate):
        self.day="00"
        self.month="00"
        self.year="0000"
        x=str(fulldate).split("/")

        if(x.__len__()==2):
         try:
            self.month=str(x[0])
            if(str(x[1]).__len__()==2):
             if(int(x[1])<30):
                self.year="20"+str(x[1])
             else:
                self.year="19"+str(x[1])
            elif(str(x[1]).__len__()==4):
                self.year=str(x[1])
         except:
             print("ERROR-DATE-MM/YY")
        elif(x.__len__()==3):
            try:
             self.day=str(x[0])
             self.month=str(x[1])
             if(str(x[2]).__len__()==2):
                if(int(x[2])<30):
                    self.year="20"+str(x[2])
                else:
                    self.year="19"+str(x[2])
             elif(str(x[2]).__len__()==4):
                self.year=str(x[2])
            except:
             print("ERROR-DATE-DD/MM/YY")

    def dict(self):
        thisdict = {
            "day": self.day,
            "month": self.month,
            "year": self.year
        }
        return thisdict
    def __str__(self):
        return str(self.year)+"-"+str(self.month)+"-"+str(self.day)

def searchlist(source,search):
  try:
   soup = BeautifulSoup(str(source), "html.parser")
   aa=[]
   for a in soup.find_all("ul",class_="profile__basics__list"):
       for z in a.descendants:
        aa.append(str(z).rstrip().lstrip())
   p=""
   for i in range(0, aa.__len__(), 1):
        if(str(aa[i]).__contains__(str(search))):
               p= str(aa[i+1])
   #print("return-"+str(p))
   return str(p)
  except:
      #print("ERROR-CANNOT FIND IN LIST-"+search)
      return ""
def searchany(source,type,classs,search):
    try:
        soup = BeautifulSoup(str(source), "html.parser")
        aa=[]
        for a in soup.find_all(str(type),class_=str(classs)):
            for z in a.descendants:
                aa.append(str(z).rstrip().lstrip())
        p=""
        for i in range(0, aa.__len__(), 1):
            if(str(aa[i]).__contains__(str(search))):
                p= str(aa[i+1])
        #print("return-"+str(p))
        return str(p)
    except:
        #print("ERROR-CANNOT FIND IN LIST-"+search)
        return ""

# finding all <li> tags

# printing the content in <li> tag



class Model:
    def __init__(self,card):
        self.gender=""
        self.name=""
        self.ethnicity=""
        self.nationality=""
        self.hair=""
        self.eye=""
        self.breasttype=""
        self.breastsize=""
        self.waist=""
        self.hips=""
        self.tattoo=""
        self.piercing=""
        self.height=""
        self.weight=""
        self.dob=Date("00/00/0000").dict()
        self.url=""
        self.rank=""
        self.phrank=""
        self.phmovies=""
        self.movies=""
        self.bio=""
        self.about=""
        soup = BeautifulSoup(str(card), "html.parser")

    def setLink(self,source):
        self.name=""
        self.movies=""
        self.breasttype=""
        self.breastsize=""
        self.waist=""
        self.hips=""
        self.gender=""
        self.hair=""
        self.dob=Date("00/00/0000")
        self.ethnicity=""
        self.nationality=""
        self.tattoo=""
        self.piercing=""

    def phuburl(self):
        return "https://www.pornhub.com/pornstar/"+self.name.lower().replace(" ","-")+"/videos"
    def downloadimages(self):
        p1="C:/Users/Dylan/Desktop/modelpics/"+self.name+"-1.jpg"
        p2="C:/Users/Dylan/Desktop/modelpics/"+self.name+"-2.jpg"
        p3="C:/Users/Dylan/Desktop/modelpics/"+self.name+"-3.jpg"

        p4="C:/Users/Dylan/Desktop/modelpicsmerged/"+self.gender+"/"+str(self.name)+".jpg"
        if(self.gender=="M"):
            try:
             save_image("https://images.psmcdn.net/tsv4/model/profiles/"+str(self.name).lower().replace(" ","_")+".jpg",p4)
            except:
                print("ERROR SAVING -"+str(self.name))


        if(self.gender=="F"):
         try:
           save_image("https://images.psmcdn.net/tsv4/model/profiles/"+str(self.name).lower().replace(" ","_")+".jpg",p1)
           save_image("https://images.psmcdn.net/tsv4/model/covers/1500/"+str(self.name).lower().replace(" ","_")+".jpg",p2)
         except:
             print("ERROR SAVING -"+str(self.name))
        #merge(p1,p2,p3)



    def dictcustom(self):

        output = {k:v for (k,v) in self.__dict__.items()}

        return output
    def __str__(self):
        return str(self.year)+"-"+str(self.month)+"-"+str(self.day)

def xnxx():


            browser = webdriver.Chrome()
            browser.set_window_size(1024, 768)
            time.sleep(1)
            for line in open("C:\\Users\\Dylan\\Desktop\\links2.txt"):
                url=str(line)
                folder=url.replace("https://forum.xnxx.com/threads/","").split(".")[0]



                test_str = ''.join(letter for letter in folder if letter.isalnum())
                try:
                 os.mkdir("C:\\Users\\Dylan\\Pictures\\"+test_str)
                 print(test_str)
                except:
                 print("----------")
                browser.get(url)
                time.sleep(1)
                source = browser.page_source
                soup = BeautifulSoup(str(source), "html.parser")
                for a in soup.find_all("span", class_="pageNavHeader"):
                        link= str(a).split(" of ")[1]
                        link2=link.split("<")[0]
                x=0
                i=1
                ii=int(link2)+1
                while(i<ii):
                 source = browser.page_source
                 soup = BeautifulSoup(str(source), "html.parser")
                 for a in soup.find_all("div", class_="messageContent"):
                    for b in a.find_all("img", class_="bbCodeImage LbImage"):
                     try:
                        link= str(b).split("src=\"")[1]
                        link2=link.split("\" ")[0]
                        x=x+1
                        print(link2)
                        try:
                         if(link2.__contains__(".gif")):
                          save_image(link2,"C:\\Users\\Dylan\\Pictures\\"+str(test_str)+"\\"+str(test_str)+'-'+str(x)+".gif")
                         else:
                          save_image(link2,"C:\\Users\\Dylan\\Pictures\\"+str(test_str)+"\\"+str(test_str)+'-'+str(x)+".jpg")
                        except:
                            print("EXCEPTION"+str(link2))
                     except:
                         print("EXCEPTION"+str(link2))
                 i=i+1
                 browser.get(url+"page-"+str(i))
                 time.sleep(1)


if __name__ == "__main__":
    teams()
