import pandas as pd
import numpy as np
import math
# Load the Training and Testing data
data=pd.read_csv("iriss.csv")
print(data.iloc[:].values)
header=list(data.columns)
header.remove("loan")

x_train=np.concatenate((data.iloc[:7817,:7].values,data.iloc[:7817,8:].values),axis=1)

y_train=data.iloc[:7817,7].values
x_test=np.concatenate((data.iloc[7817:10162,:7].values,data.iloc[7817:10162,8:].values),axis=1)
y_test=data.iloc[7817:10162,7].values
x1_test=np.concatenate((data.iloc[10162:,:7].values,data.iloc[10162:,8:].values),axis=1)
y1_test=data.iloc[10162:,7].values
# class node for building the decision tree
class node:
     pass


# Entropy function to calculate the entropy
def Entropy(p,n):
    if p==0 or n==0:
        return 0
    p1=p/(p+n)
    n1=n/(p+n)
    en=(p1)*(-1)*math.log(p1,2)+n1*(-1)*math.log(n1,2)
    return en


# Function for counting number of positive and negative example in dataset
def count(y):
    p=0
    n=0
    for i in range(len(y)):
        if y[i]=="yes":
            p+=1
        else:
            n+=1
    return p,n


# Function for checking, attribute is numeric or not
def isstring(x):
    try:
        v=int(x)
        return False
    except:
        v=str(x)
        return True


# Function for counting the values and examples on a attribute
def count_attribut(x,y):
    dict={}
    dicp={}

    for i in range(len(x)):
        if dict.get(x[i])==None:
            dict[x[i]]=1
            if(y[i]=="yes"):
                dicp[x[i]]=1
        else:
            dict[x[i]]+= 1
            if (y[i] == "yes"):
                if dicp.get(x[i])==None:
                    dicp[x[i]]= 1
                else:
                    dicp[x[i]]+=1
    return dict,dicp


# Function for calculating the median
def median(x):
    x=sorted(x)
    if len(x)%2==0:
        return (x[int(len(x)/2)+1]+x[int(len(x)/2)])/2
    else:
        return x[int(len(x)/2)]


# Function for calculating the information gain for a attribute
def infoGain(x,y,cn):
    p,n=count(y)
    s=Entropy(p,n)
    ln=len(y)

    if(isstring(x[0][cn])):
        sum=0
        dict,dicp=count_attribut(x[:,cn].copy(),y.copy())

        for i in dict.keys():
            if dicp.get(i)==None:
                p=0
                n1=dict[i]
            else:
                p=(dicp[i])/dict[i]
                n1=(dict[i]-dicp[i])/dict[i]
            sum+=Entropy(p,n1)*dict[i]/ln
        sum=s-sum;
        li=[]
        li.append(float(sum))
        li.append(len(dict.keys()))
        return li
    else:
        m=median(x[:,cn].copy())
        lp=[0,0]
        rp=[0,0]
        for i in range(len(y)):
            if(x[i][cn]<m):
                if y[i]=="yes":
                    lp[0]+=1
                else:
                    lp[1]+=1
            else:
                if y[i]=="yes":
                    rp[0]+=1
                else:
                    rp[1]+=1
        tlp=(lp[0]+lp[1])/ln
        trp=(rp[0]+rp[1])/ln
        sum=(s-(tlp*Entropy(lp[0],lp[1])+trp*Entropy(rp[0],rp[1])))
        li=[]
        li.append(float(sum))
        li.append(-1)
        return li


# Function for distributing the examples on their children in the case of numerical attribute
def make_tableR(x,y,c,la):
    x1=np.ndarray(0)
    y1=np.ndarray(0)
    if(c==0):
        m = median(x[:, la].copy())
        for i in range(len(x)):
            if(x[i][la]<m):
                if x1.size == 0:
                    x1 = np.hstack(x[i, :])
                    x1 = np.vstack((x1, x[i, :]))
                    x1 = np.delete(x1, (0), axis=0)
                    y1 = np.append(y1, y[i])
                else:
                    x1 = np.vstack((x1, x[i, :]))
                    y1 = np.append(y1, y[i])
    elif (c == 1):
        m = median(x[:, la].copy())
        for i in range(len(x)):
            if (x[i][la] >=m):
                if x1.size == 0:
                    x1 = np.hstack(x[i, :])
                    x1 = np.vstack((x1, x[i, :]))
                    x1 = np.delete(x1, (0), axis=0)
                    y1 = np.append(y1, y[i])
                else:
                    x1 = np.vstack((x1, x[i, :]))
                    y1 = np.append(y1, y[i])
    return x1,y1


# Function for distributing the examples on their children in the case of non_numeric attribute
def make_tableC(x,y,lab,l):
    x1=np.ndarray(0)
    y1=np.ndarray(0)
    for i in range(len(x)):
        if x[i][l]==lab:
             if x1.size==0:
                 x1=np.hstack(x[i,:])
                 x1=np.vstack((x1,x[i,:]))
                 x1=np.delete(x1,(0),axis=0)
                 y1=np.append(y1,y[i])
             else:
                 x1=np.vstack((x1,x[i,:]))
                 y1=np.append(y1,y[i])
    return x1,y1


# Main function for building the tree
def build_tree(x_train,y_train,he):
    p,n=count(y_train)
    if(len(he)==0):

        h1=node()
        h1.label="null"
        h1.pn=[]
        h1.branch=[]
        h1.pn.append(p)
        h1.pn.append(n)
        h1.leaf=True
        if(p>=n):
            h1.leafv="yes"
        else:
            h1.leafv="no"
        return h1
    elif p<4 or n<6:
        h1 = node()
        h1.pn = []
        h1.branch=[]
        h1.pn.append(p)
        h1.pn.append(n)
        h1.label = "null"
        h1.leaf = True
        if (p >= n):
            h1.leafv = "yes"
        else:
            h1.leafv = "no"
        return h1
    gain=[]
    for i in range(len(he)):
        gain.append(list(infoGain(x_train.copy(),y_train.copy(),i)))
    l=[0.0000,0]
    la=0.0000000000
    labe=0
    for i in range(0,len(gain)):
        if la<=float(gain[i][0]):
            l[0]=gain[i][0]
            l[1]=gain[i][1]
            la=l[0]
            labe=i
    if (l[1] == 1) or l[1] == 0 :
        h1 = node()
        h1.pn = []
        h1.branch=[]
        h1.pn.append(p)
        h1.pn.append(n)
        h1.label = "null"
        h1.leaf = True
        if (p >= n):
            h1.leafv = "yes"
        else:
            h1.leafv = "no"

        return h1

    nw=node()
    nw.branch=[]
    nw.addr=[]
    nw.leaf=False
    nw.leafv="null"
    nw.pn=[]
    p,n=count(y_train)
    nw.pn.append(p)
    nw.pn.append(n)
    nw.label=he[labe]
    if l[1]!=-1:
        dict, dicp = count_attribut(x_train[:, labe].copy(), y_train.copy())
        hade = list(dict.keys())
    he.remove(he[labe])

    if l[1]==-1:
        for i in range(2):
            nw.branch.append(median(x_train[:,labe].copy()))
            x_train1,y_train1=make_tableR(x_train,y_train,i,labe)
            if x_train1.size!=0:
                x_train1 =(np.concatenate((x_train1[:, :labe], x_train1[:, labe + 1:]), axis=1))
            nw.addr.append(build_tree(x_train1.copy(),y_train1.copy(),he.copy()))
        return nw
    else:
        for i in range(l[1]):
            x_train1,y_train1=make_tableC(x_train,y_train,hade[i],labe)
            nw.branch.append(hade[i])
            if x_train1.size!=0:
                x_train1 =(np.concatenate((x_train1[:, :labe], x_train1[:, labe + 1:]), axis=1))
            nw.addr.append(build_tree(x_train1.copy(),y_train1.copy(),he.copy()))
        return nw


# Function for giving the index of label
def Give_Hnumb(st,hd):
    for i in range(len(hd)):
        if(st==hd[i]):
            return i


# Function for calculating the accuracy
def accuracy(x,y):
    c=0
    for i in range(len(x)):
        if x[i]==y[i]:
            c+=1
    return(c/len(x))*100


# Function for testing the data on tree
def TestPhase(x_test,y_test,hn,hv,head,hd,v11):
    predict=np.ndarray(0)
    for i in range(len(x_test)):
        h=head
        while(h.leaf==False):
            index=Give_Hnumb(h.label,hd)
            c=0
            co=0
            if(isstring(x_test[i][index])):
                for j in range(len(h.branch)):

                    if x_test[i][index]==h.branch[j]:

                        if(hv!="null"):
                            if(h.addr[j]==hn):
                                c=1
                                break
                        h=h.addr[j]
                        co=1
                        break;
                if co==0:
                    co=-1
                    break;
            else:
                if x_test[i][index]<h.branch[0]:
                    if (hv != "null"):
                        if (h.addr[0] == hn):
                            c = 1
                            break
                    h=h.addr[0]
                else:
                    if (hv != "null"):
                        if (h.addr[1] == hn):
                            c = 1
                            break
                    h=h.addr[1]
        if c==1:
            predict=np.append(predict,hv)
        elif co==-1:
            if h.pn[0]>h.pn[1]:
                va="yes"
            else:
                va="no"
            predict=np.append(predict,va)
        else:
            predict=np.append(predict,h.leafv)
    if v11==1:
        return accuracy(predict,y_test)
    elif v11==0:
        return accuracy(predict,y_test),predict


# Function, checks that the node has all branches leaf or not
def Notleaff(n1):
    for i in range(len(n1.branch)):
        n2=n1.addr[i]
        if(n2.leaf==False):
            return True
    return False


# Function that use in pruning to find the node , that has all branches leaf, and their parent
def last_subtree(n2,j):
    n1=n2
    nw= n1
    n1= n1.addr[j]
    while(Notleaff(n1)==True):
        nw=n1
        for i in range(len(n1.branch)):
            n2 = n1.addr[i]
            if (n2.leaf == False):
                n1 = n1.addr[i]
                break;
    return nw,n1


# Function for pruning the tree
def Post_Pruning(h,x_test,y_test,head):
    n1=h
    n2=h
    for j in range(0,len(n1 .branch)):
        n2=h
        pre,ch=last_subtree(n2,j)
        if pre.pn[0]>pre.pn[1]:
            v="yes"
        else:
            v="no"

        while(TestPhase(x_test,y_test,None,"null",h,head,1)<=TestPhase(x_test,y_test,ch,v,h,head,1)):

            ch.leaf=True
            ch.leafv=v

            ch.branch=[]
            ch.addr=[]

            pre,ch=last_subtree(n2,j)
            if ch.label==n2.addr[j].label:
                break

            if pre.pn[0] > pre.pn[1]:
                v = "yes"
            else:
                v = "no"
    return n1


# Function for splitting the dataset for k-fold cross_validation
def split(x_train,y_train,c):
    if(c==0):
        x_cross=x_train[:469]
        y_cross=y_train[:469]
        x_train1=x_train[469:]
        y_train1 = y_train[469:]
    else:
        x_cross = x_train[c:c+469]
        y_cross = y_train[c:c+469]
        x_train1 = np.concatenate((x_train[:c,:],x_train[c+469:,:]),axis=0)
        y_train1 = np.concatenate((y_train[:c],y_train[c+469:]),axis=0)
    return x_cross,y_cross,x_train1,y_train1


# Function for k-fold cross validation
def K_fold(x_train,y_train,x1_test,y1_test):
    c=0
    acc=[]
    nw=node()
    nw.addr=[]
    while(c!=2345):
        x_cross,y_cross,x_train1,y_train1=split(x_train.copy(),y_train.copy(),c)
        hed = list(data.columns)
        hed.remove("loan")
        n1=build_tree(x_train1.copy(),y_train1.copy(),hed.copy())

        h1=Post_Pruning(n1,x_cross,y_cross,hed.copy())
        acc.append(TestPhase(x1_test.copy(),y1_test.copy(),None,"null",h1,hed.copy(),1))
        nw.addr.append(h1)

        c+=469
    c=0.0
    index=0;
    print("Accuracy list for different different tree that form using k-fold cross validation is: ")
    print(acc)
    for i in range(len(acc)):
        if c<=acc[i]:
            c=acc[i]
            index=i
    return nw.addr[index]





#Main part
if __name__=="__main__":
    print("                      ********** ID3 ALGORITHM ********** ")
    head=K_fold(x_train,y_train,x1_test,y1_test)
    acc,predict=TestPhase(x_test,y_test,None,"null",head,header.copy(),0)
    print(" Predict values on Testing data set are: ")
    print(predict)
    print(" True value of Testing data set are: ")
    print(y_test)
    print("Accuracy on testing dataset is: ")
    print(acc)


















