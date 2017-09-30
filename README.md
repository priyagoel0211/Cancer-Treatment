# Cancer-Treatment
memory.limit(56000)
library(plyr)
library(dplyr)
library(ggplot2)
library(tidytext)
library(tidyr)
library(readr)
library(stringi)
library(stringr)
library(tibble)
library(tm)
library(SnowballC)
library(wordcloud)
library(caret)
library(grid)
library(gridExtra)
library(randomForest)
library(rpart)
library(e1071)

setwd("C:/Users/PRIYA/Desktop/Kaggle")
training_var=read.csv("training_variants.csv")

test_var=read.csv("test_variants.csv")
# Importing the text data
train_txt_dump <- tibble(text = read_lines('training_text', skip = 1))
train_txt= separate(train_txt_dump, text, into = c("ID", "clinical_txt"), sep = "\\|\\|")
train_txt$ID =as.integer(train_txt$ID)

test_txt_dump <- tibble(text = read_lines('test_text', skip = 1))
test_txt= separate(test_txt_dump, text, into = c("ID", "clinical_txt"), sep = "\\|\\|")
test_txt$ID =as.integer(test_txt$ID)

# for the purpuse of preserving memory in R, we fill time to time delete the data bases which are are not required further
rm(train_txt_dump,test_txt_dump)

# Glimse of train variant data
head(training_var)
str(training_var)
training_var$Class=as.factor(training_var$Class)
str(training_var)
summary(training_var,maxsum=9)

#checking out the missing value
sum(is.na(training_var))

#Variant Data exploration
Var.freq = sort(table(training_var$Variation), decreasing = T)
summary(Var.freq)
head(Var.freq)

# Gene data exploration
Gene.freq = sort(table(training_var$Gene), decreasing = T)
summary(Gene.freq)
head(Gene.freq,15)

# Class data exploration
Class.freq = sort(table(training_var$Class), decreasing = T)
summary(Class.freq)
head(Class.freq,15)

# Text Analysis: To analyse the text data, we will first make it handy. 
#For this purpuse the prepocessing of text data is required. 
#The preprocessing  should be done in the same way for train text and test text. 
#Hence, the merging of both data set is requied first. 
#For this purpose, we will first merge variant data with text data for both and then will merge train and test.
train=merge(training_var,train_txt, by = "ID")
test=merge(test_var,test_txt, by = "ID")

full_data=full_join(train,test)
sum(is.na(full_data$Class))

#Strart preprocessing:converting the text into corpus to create a bag of words
set.seed(123)
corpus = VCorpus(VectorSource(full_data$clinical_txt))
corpus = tm_map(corpus,content_transformer(stringi::stri_trans_tolower))
corpus = tm_map(corpus,removePunctuation,preserve_intra_word_dashes=T)
corpus = tm_map(corpus, removeWords, stopwords("english"))

# I have created a  my own stopword list
corpus = tm_map(corpus,removeWords,c("author","describe","find","found", "result",
                                     "conclude","analyze","analysis","show","shown","resulted","concluded","described",
                                     "concluded","evaluate","evaluated","discuss","discussed","demonstrate","demonstrated",
                                     "the","this","that","these","those","illustrated","illustrate","list","fig","figure",
                                     "et","al","data","determined","studied","indicated","research","method","determine",
                                     "studies","study","indicate","research","researcher","medical","background","abstract",
                                     "and","but","all","also","are","been","both","can","consider","describe","described",
                                     "declar","determin","did","rt","http\\*"))
txt = tm_map(corpus, stemDocument, language="english")
txt <- tm_map(corpus, stripWhitespace)

# Converting the corpus into Sparse Matrix for the purpose of analysis
dtm <- DocumentTermMatrix(corpus)
dtm
# No. of terms in the term matrix are 275892, which actually represent the no. of columns in the term matrix, which is actually very high. There may exist some very few frequency words.
dtm = removeSparseTerms(dtm, sparse = 0.95) 
dtm
#Glipse of Data
txt_data <- as.matrix(dtm)
str(txt_data)

# Identifying most frequest words and their frequence
count_word <- colSums(txt_data)
length(count_word)
fre_word<- tibble(name = attributes(count_word)$names, count = count_word)
top_n(fre_word,10)
top_n(fre_word,-10)
wordcloud(names(count_word), count_word, min.freq = 10000,scale = c(6,.1), colors = brewer.pal(6, 'Dark2'))

#Dimension Reduction using PCA
install.packages("e1071")
library(e1071)
set.seed(123)
pca=preProcess(x=txt_data,method = "pca", thresh = 0.60)
text_predictors_pca=predict(pca,txt_data)
View(text_predictors_pca)
ncol(text_predictors_pca)

#Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = full_data[,2:3],fullRank = T)
var_predictors <- data.frame(predict(dmy, newdata = full_data[,2:3]))
str(var_predictors)
ncol(var_predictors)
newdata=cbind(var_predictors,text_predictors_pca)
ncol(newdata)
newdata$Class=as.factor(full_data$Class)
newdata$ID=as.integer((full_data$ID))
summary(newdata)
str(newdata$Class)
sum(is.na(newdata))

#Splitting the data into train and test
train_newdata = newdata[1:3321,]
nrow(train_newdata)
sum(is.na(train_newdata$Class))
test_newdata = newdata[3322:8989,]
nrow(test_newdata)
test_newdata$Class=NULL
rm(corpus,count_word,dmy,dtm,fre_word,full_data,pca,text_predictors_pca,txt,txt_data,var_predictors)

#Selecting the best features
set.seed(123)
rf1=randomForest(data=train_newdata[,-10452],Class~., ntree=1000)
Var_imp <- importance(rf1)
Varimp=as.data.frame(Var_imp)
subset=subset(Varimp,Varimp$MeanDecreaseGini>=1)
subset=as.matrix(subset)
rownames(subset)
best_feature_subset=train_newdata[c(rownames(subset),]
View(best_feature_subset)

#Now we are ready with our data to be train for classification 
best_feature_subset$Class=train_newdata$Class
View(best_feature_subset)
str(best_feature)
class(best_feature_subset$Class)
sum(is.na(best_feature_subset))
# Let's train the model

#Creating grid
grid = expand.grid(nrounds=c(500,800,1000),
                    max_depth=c(6,8,10),
                    eta = c(0.1,0.2,03),
                    gamma=0.1,
                    colsample_bytree=c(0.4,0.6,0.8,1),
                    min_child_weight=1, 
                    subsample=0.6)
grid
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
#training the model with default parameters
model_xgb=train(x = best_feature_subset[,-501],y = best_feature_subset$Class,
                method = "xgbTree")
model_xgb
predictions<-predict.train(object=model_xgb,test_newdata,type="raw")

#Confusion Matrix
cm = confusionMatrix(predictions,test_newdata$outcome)
cm
