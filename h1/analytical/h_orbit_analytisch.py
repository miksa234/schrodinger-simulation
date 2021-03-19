from scipy.constants import physical_constants
import numpy as np
from scipy.special import assoc_laguerre as lag #zugehörige Laguerre Polynome
from scipy.special import sph_harm as kugelfl #Kugelflächenfunktionen
import matplotlib.pyplot as plt

(a0,blub,bla)=physical_constants["Bohr radius"]
a0=0.529177210903 #in Angsröm

def fak(n):
  #Beschreibung: Diese Funktion ermittelt zur natürliche Zahl n die zugehörige Fakultät (n!)
  #                 Variablenname:      Datentyp:       Beschreibung:
  #Input:           n                   integer         Zu Berechnende Fakultät. ACHTUNG: es gilt n>=0 und n element N
  #Output:          -                   integer         n!
  #Fehleroutput                         integer=-1      Falsche Eingabe für n

  if (n<0) or (n-int(n)>0):
    print("Upps, das hätte leider nicht passieren dürfen.\nn muss eine natürliche Zahl sein.")
    return -1
  elif (n==0) or (n==1):
    return 1
  else:
    return n*fak(n-1)

def transform(x,y,z):
  # Beschreibung: Diese Funktion Transfromiert Kartesische Koordinaten in Kugelkoordinaten
  #                  Variablenname:      Datentyp:       Beschreibung:
  # Input:           x                   real            kartesische x Koordinate
  #                  y                   real            kratesische y Koordinate
  #                  z                   real            kartesische z koordinate
  # Output:          r                   real            Radius
  #                  theta               real            Polarwinkel mit theta\in[0,pi]
  #                  phi                 real            Azimutwinkel mit phi\in[0,2*pi]

  # Berechnung von r
  r=np.sqrt(x**2+y**2+z**2)
  #Problembehandlung: Koordinatenursprung
  if(r==0):
    phi=0.
    theta=0.
  else:
    #Berechnung von theta
    theta=np.arccos(z/r)

    # Berechnung von phi
    if (x>0)and(y>=0):
      phi=np.arctan(y/x)
    elif(x==0)and(y>0):
      phi=np.pi/2
    elif(x<0):
      phi=np.arctan(y/x)+np.pi
    elif(x==0)and(y<0):
      phi=(3/2)*np.pi
    elif(x>0)and(y<0):
      phi=2*np.pi+np.arctan(y/x)
    else:#if(x==0)and(y==0)
      phi=0.
  return r,theta,phi

def norm(Z,n,l):
  #Beschreibung: Diese Funktion ermittelt den Normierungsfaktor der Wellenfunktion für ein Elektron um einen Atomkeern der Ladung Z*e
  #                 Variablenname:      Datentyp:       Beschreibung:
  #Input:           Z                   integer         Anzahl der Protonen im Atomkern
  #                 n                   integer         Hauptquantenzahl
  #                 l                   integer         Nebenquantenzahl
  #Output:          -                   real            Normierungsfaktor der Wellenfunktion

  return np.sqrt((((2*Z)/(n*a0))**3)*(fak(n-l-1)/(2*n*fak(n+l))))

def radial(Z,n,l,r,normierung):
  #Beschreibung: Diese Funktion ermittelt den Radialen Anteil R_nl(r) der Wellenfunktion für das Wasserstoffatom
  #                 Variablenname:      Datentyp:       Beschreibung:
  #Input            Z                   integer         Anzahl der Protonen im Atomkern
  #                 n                   integer         Hauptquantenzahl
  #                 l                   integer         Nebenquantenzahl
  #                 r                   real            Abstand des Elektrons zum Atomkern
  #                 normierung          real            Normierungsfaktor der Wellenfunktion
  #Output:          -                   real            Radialer Anteil R_nl(r) der Wellenfunktion für das Wasserstoffatom

  rho=(2*Z*r)/(n*a0)
  return normierung*np.exp(-rho/2)*(rho**l)*lag(rho,n-l-1,2*l+1)

def aufenthalt_radial(Z,n,l,r,normierung):
  #Beschreibung: Diese Funktion ermittelt die Radiale Aufenthaltswahrscheinlichkeitsdichte des Wasserstoffatoms
  #                 Variablenname:      Datentyp:       Beschreibung:
  #Input            Z                   integer         Anzahl der Protonen im Atomkern
  #                 n                   integer         Hauptquantenzahl
  #                 l                   integer         Nebenquantenzahl
  #                 r                   real            Abstand des Elektrons zum Atomkern
  #                 normierung          real            Normierungsfaktor der Wellenfunktion
  #Output:          -                   real            Radiale Aufenthaltswahrscheinlichkeitsdichte des Wasserstoffatoms

  return np.sqrt(radial(Z,n,l,r,normierung)**2)**2

def aufenthalt_welle(Z,n,l,m,r,normierung,phi,theta):
  #Beschreibung: Diese Funktion ermittelt die Radiale Aufenthaltswahrscheinlichkeitsdichte des Wasserstoffatoms
  #                 Variablenname:      Datentyp:       Beschreibung:
  #Input            Z                   integer         Anzahl der Protonen im Atomkern
  #                 n                   integer         Hauptquantenzahl
  #                 l                   integer         Nebenquantenzahl
  #                 m                   integer         magnetische Quantenzahl des Drehimpulses
  #                 r                   real            Abstand des Elektrons zum Atomkern
  #                 normierung          real            Normierungsfaktor der Wellenfunktion
  #                 phi                 real            Azimutwinkel mit phi\in[0,2*pi]
  #                 theta               real            Polarwinkel mit theta\in[0,pi]
  #Output:          result              real            Aufenthaltswahrscheinlichkeitsdichte des Wasserstoffatoms

  result=radial(Z,n,l,r,normierung)*kugelfl(m,l,phi,theta)
  return np.sqrt(result.conjugate()*result)**2


def plotaufenthalt_radial(Z,n,l,aufl,name_nl,xmax):
  #Beschreibung: Diese Funktion plottet die Radialenaufenthaltswahrscheinlichkeitsdichte*r**2, wobei nach Funktionsaufruf plt.show() ausgeführt werden muss
  #                 Variablenname:      Datentyp:           Beschreibung:
  #Input            Z                   integer             Anzahl der Protonen im Atomkern
  #                 n                   integer,array       Hauptquantenzahl, das Array muss dieselbe Größe wie l und name_nl haben
  #                 l                   integer,array       Nebenquantenzahl, das Array muss dieselbe Größe wie n und name_nl haben
  #                 aufl                integer             Auflösung des Graphen
  #                 name_nl             characther,array    Legende des Plots, das Array muss dieseleb Größe wie n und l haben
  #                 xmax                real                maximaler Abstand des Elektrons zum Atomkern, wobei innerhald der Funktion xmax in xmax*a0 umgerechnet wird
  #Output:          -                   -                   Plotvorbereitung für die Radialenaufenthaltswahrscheinlichkeitsdichte*r**2

  x=np.linspace(0,xmax,aufl)
  x=x*a0
  anz=np.shape(n)[0] #Anzahl der zu plottenden Orbitale
  y=np.zeros((anz,aufl))
  r=x
  for i in range(0,anz):
    normi=norm(Z,n[i],l[i])
    y[i,:]=aufenthalt_radial(Z,n[i],l[i],r[:],normi)*r**2

  # Plot
  plt.figure()
  for i in range(0,anz):
    plt.plot(x,y[i,:],label=name_nl[i])
  plt.title("Radiale Aufenthaltswahrscheinlichkeitsdichte *r²")
  plt.ylabel("|R_nl(r)|*r²")
  plt.xlabel("r*a0 [Angström]")
  plt.legend()

def plot2d_aufenthalt(Z,n,l,m,aufl,xmin,xmax,ymin,ymax,name):
  #Beschreibung: Diese Funktion plottet die x-z Ebene der Aufenthaltswahrscheinlichkeitsdichte und die Aufenthaltswahrscheinlichkeitsdichte*r**2,
  #              wobei nach Funktionsaufruf plt.show() ausgeführt werden muss
  #                 Variablenname:      Datentyp:           Beschreibung:
  #Input            Z                   integer             Anzahl der Protonen im Atomkern
  #                 n                   integer,array       Hauptquantenzahl, das Array muss dieselbe Größe wie l und name haben
  #                 l                   integer,array       Nebenquantenzahl, das Array muss dieselbe Größe wie n und namehaben
  #                 aufl                integer             Auflösung
  #                 xmin,xmax           real                Gittergrenzen in x Richtung
  #                 ymin,ymax           real                Gittergrenzen in y Richtung
  #                 name                character,array     enthält die Spezifikation von n,l und m die geplottet werden und so in der Überschrift angezeigt werden,
  #                                                         das Array muss dieselbe Größe wie n und l haben
  #Output:          -                   -                   Plotvorbereitung für die Aufenthaltswahrscheinlichkeitsdichte und die Aufenthaltswahrscheinlichkeitsdichte*r**2

  #Definiere Gitter, Achtung y entspricht hier den Werten auf der z Achse
  x=np.linspace(xmin,xmax,aufl)*a0
  y=np.linspace(ymin,ymax,aufl)*a0
  X,Y=np.meshgrid(x,y)

  blub=np.shape(X)#Form von np.shape=(Anzahl y Werte, Anzahl x Werte, Anzahl z Werte)

  #Bereite Transformation in Kugelkoordinaten vor
  r=np.zeros(blub);theta=np.zeros(blub);phi=np.zeros(blub)
  #Koordinatentransformation in Kugelkoordinaten
  for i in range(0,blub[0]): #Schleife für z Werte
    for j in range(0,blub[1]): #Schleife für x Werte
      (r[i,j],theta[i,j],phi[i,j])=transform(X[i,j],0,Y[i,j])

  #Plottvorbereitungen
  anz=np.shape(n)[0] #Anazhl der zu Plottenden Orbitale

  for number in range(0,anz):
    #Berechne Normaisierungsfaktor
    normi=norm(Z,n[number],l[number])
    #Definiere Plot Array
    orbit=np.zeros(np.shape(X))
    #Berechne Raumpunkte
    orbit=np.real(aufenthalt_welle(Z,n[number],l[number],m[number],r,normi,phi,theta))

    #Plot
    #Plot Aufenthaltswahrscheinlichkeitsdichte
    plt.figure()
    plt.imshow(orbit,cmap="gnuplot",extent=[x.min(),x.max(),y.min(),y.max()])
    plt.title("Aufenthaltswahrscheinlichkeitsdichte\nfür "+name[number])
    plt.xlabel("[Angström]")
    plt.ylabel("[Angström]")
    plt.colorbar()
    #Plot Aufenthaltswahrscheinlichkeitsdichte*r²
    plt.figure()
    plt.imshow(orbit*r**2,cmap="gnuplot",extent=[x.min(),x.max(),y.min(),y.max()])
    plt.title("Aufenthaltswahrscheinlichkeitsdichte*r²\n für "+name[number])
    plt.xlabel("[Angström]")
    plt.ylabel("[Angström]")
    plt.colorbar()


#Plot der Radialenaufenthaltswahrscheinlichkeitsdichte*r**2
Z=1 #Anzahl der Protonen im Kern
n=np.array([1,2,2,])
l=np.array([0,0,1])
aufl=200 #Auflösung für den Plot der Radialenaufenthaltswahrscheinlichkeitsdichte*r**2
name_nl=np.array(["n=1,l=0","n=2,l=0","n=2,l=1"]) #Legende des Plots
xmax=20

plotaufenthalt_radial(Z,n,l,aufl,name_nl,xmax)


#Plot x-z Ebene
Z=1
n=np.array([1,2,2])
l=np.array([0,0,1])
m=np.array([0,0,0])
#Definiere Gitterauflösung
aufl=403
#Definiere Gittergrenzen
xmin=-20
xmax=20
ymin=-20
ymax=20
name=np.array(["n=1, l=0, m=0","n=2, l=0, m=0","n=2, l=1, m=0"]) #Quantenzahlen, die in der Überschrift angezeigt werden

plot2d_aufenthalt(Z,n,l,m,aufl,xmin,xmax,ymin,ymax,name)

plt.show()
