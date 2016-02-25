from mnist import load_mnist
import numpy
from preprocessing import deskew
from sklearn.metrics import accuracy_score
[feature,lable]=load_mnist("training")
[tfeature,tlable]=load_mnist("testing")
for i in range(0,60000):
    temp=deskew(feature[i,:])
temp =feature.reshape(60000,28*28)
templ = numpy.squeeze(numpy.asarray(lable))
ttemp=tfeature.reshape(10000,28*28)
ttempl=numpy.squeeze(numpy.asarray(tlable))

smalldata=temp[0:20000,:]
smalllable=numpy.squeeze(numpy.asarray(lable[0:20000,:]))
print 'preprocessing finished'
#print smalldata.shape
#print smalllable.shape
#print
#print templ
#print templ.shape
from sklearn import svm
from sklearn.externals import joblib

print "Training"
clf=svm.SVC(degree=3,kernel='poly',decision_function_shape='ovr')
clf.fit(temp,templ)
print "Train finished"
joblib.dump(clf, 'poly.pkl')
res=clf.predict(ttemp)
acc=accuracy_score(ttempl,res)
print acc