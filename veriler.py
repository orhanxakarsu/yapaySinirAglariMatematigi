import math as m
import random
import numpy as np
#from fonksiyonlar import sigmoidFonksiyonu,toplamFonksiyonu,np,reLU

def fonksiyon(sayi):
    x=[]
    y=[]
    for i in range(sayi):
        x.append([i])
        y.append([1/(1+np.exp(-sayi))])
    return x,y

class Sigmoid():
    
    def ileri(self,deger):
        a= 1/(1+(np.exp(-deger)))
        return a
    
    def geri(self,deger):
        b = deger*(1-deger)
        return b

        



class RELU():
    
    def ileri(self,deger):
        return max(0,deger)
    
    def geri(self,deger):
        return 1 if deger>0 else 0



def matrisCarpimi(vektor1,vektor2):
    if len(vektor1) !=len(vektor2):
        raise ValueError(f'vektor 1 boyutu :{len(vektor1)}, vektor 2 boyutu :{len(vektor2)}')
    else:
        toplam =0
        for i in range(len(vektor1)):
            toplam+=vektor1[i]*vektor2[i]
        return toplam
    


def VektorCarpimi(vektor1,vektor2):
    if len(vektor1) !=len(vektor2):
        raise ValueError(f'vektor1 boyutu :{len(vektor1)}, vektor2 boyutu :{len(vektor2)}')
    else:
        return [vektor1[i]*vektor2[i] for i in range(len(vektor1))]
    
    



def reLU(z):
    return max(0,z)





class katman:
    
    def __init__(self,katmandakiNoronlar,bulunduguKatman):
        
        self.katmandakiNoronlar=katmandakiNoronlar
        
        self.bulunduguKatman=bulunduguKatman
        



class Noron():
        def __init__(self,agirlik=None): 
            self.sabit=0
            self.w =agirlik if agirlik!=None else random.randint(0, 100)/100
            
        
        def toplamFonksiyonu(self,dizi,agirlik,bias):
            array = np.array(dizi)
            ayarlanmisArray =agirlik*array + bias
            return ayarlanmisArray.sum()
        
        def ileri(self,girisDegerleri,aktivasyonFonksiyonu):
            self.aktivasyonFonksiyonu=aktivasyonFonksiyonu.ileri
            #Ağırlığımız 0-1 arasına sıkıştırdık :

            #Gelen bilgileri topladığımız yer.
            self.toplam = self.toplamFonksiyonu(girisDegerleri,self.w,self.sabit)
            self.cikis = self.aktivasyonFonksiyonu(self.toplam)
            return self.cikis
        
            
            
            

class AgYapisi():
    #Ağ yapısı class'ının yapıcı fonksiyonu :
    def __init__(self,katmanAktivasyonFonksiyonlari,elemanSayilari =[]):
        
        self.katmanAktivasyonFonksiyonlari=katmanAktivasyonFonksiyonlari

        self.elemanSayilari=elemanSayilari
        
        self.katmanlardanCikanDegerler = [[0 for i in range(sayi)] for sayi in elemanSayilari]
        #Biz giriş değeri olarak katmandaki nöron sayılarını veriyoruz. Kendisi katman sayısını çıkarıyor.
        self.katmanSayisi=len(elemanSayilari)
        #Duyarlılık değerlerini bu diziye vereceğiz.
        self.duyarliliklar =[[0 for i in range(sayi)] for sayi in elemanSayilari]
        self.agYapisi =[]
        for katman in range(len(elemanSayilari)):
            random.seed(katman)
            self.ag =[]
            for i in range(int(elemanSayilari[katman])):
                self.ag.append(Noron())
            self.agYapisi.append(self.ag)
    
    def ileriYayilim(self,giris,istenilenCikis=None):
        
        
        for i,katman in enumerate(self.agYapisi):
            temp =[]
            for noron in katman:
                temp.append(noron.ileri(giris,self.katmanAktivasyonFonksiyonlari[i]))
            giris =[]
            giris=temp
            self.katmanlardanCikanDegerler[i]=giris
            temp =[]
        if istenilenCikis==None:    
            return giris
        else:
            self.hata = np.array(giris)-np.array(istenilenCikis)
            #print(self.hata)
            return giris
            
    
    
                                  
    
    
    def geriYayilim(self,X,Y,ogrenmeOrani=0.1,epochSayisi=1):
        for e in range(epochSayisi):
            for tur in range(len(X)):
                x=X[tur]
                y=Y[tur]
                self.ogrenmeOrani=ogrenmeOrani
                self.epochSayisi=epochSayisi
                tahmin =self.ileriYayilim(x,y)
                self.modelAgirliklari =[[noron.w for noron in katman] for katman in self.agYapisi]
        
        
                vektor1 =[self.katmanAktivasyonFonksiyonlari[-1].geri(i) for i in tahmin]
                
                carpilmisVektorler = VektorCarpimi(vektor1,self.hata)
                #print(carpilmisVektorler)
                self.duyarliliklar[-1]=list(-2*np.array(VektorCarpimi(vektor1,self.hata)))
                #print('a')
                #print(self.duyarliliklar[-1])
                #print(self.modelAgirliklari[-1])
                for i in range(len(self.duyarliliklar)-2,-1,-1):
                    # print(i)
                    # #print(len(self.modelAgirliklari[i+1]),len(self.duyarliliklar[i+1]))
                    # print('normal :')
                    # print(np.array(self.modelAgirliklari[i+1]).transpose())
                    # print('normal degil :')
                    # print(np.array(self.modelAgirliklari[i+1]))
                    self.duyarliliklar[i] = list(np.array([self.katmanAktivasyonFonksiyonlari[i].geri(a) for a in self.katmanlardanCikanDegerler[i]])*(matrisCarpimi(self.modelAgirliklari[i+1],self.duyarliliklar[i+1])))
        
                
                #Ağırlıkları Ayarla :
                
                for i in range(len(self.agYapisi)):
                    agirliklar = [self.agYapisi[i][j].w for j in range(len(self.agYapisi[i]))]
                    #print(i)
                    cikarilacakAgirliklar = list(self.ogrenmeOrani*np.array(VektorCarpimi(self.katmanlardanCikanDegerler[i],self.duyarliliklar[i]))) if i !=0 else list(self.ogrenmeOrani*np.array(VektorCarpimi(x,self.duyarliliklar[i])))
                    for j in range(len(self.agYapisi[i])):
                        self.agYapisi[i][j].w+= cikarilacakAgirliklar[j]
                        #print(cikarilacakAgirliklar[j])
                        self.agYapisi[i][j].sabit+=self.ogrenmeOrani*self.duyarliliklar[i][j]
                


        




def main():
    a = RELU()
    print(a.geri(5))
    noronSayilari=[1,5,5,1]
    aF=[Sigmoid(),Sigmoid(),Sigmoid(),Sigmoid()]
    yapi =AgYapisi(aF,noronSayilari)
    
    #print(yapi.ileriYayilim([622,112,100,1121]))
    
    

    X,Y=fonksiyon(10000)
    #print(Y[-1])
    yapi.geriYayilim(X,Y,epochSayisi=10)
    #print(yapi.agYapisi)
    print(yapi.hata)
    

if __name__=='__main__' :
    main()