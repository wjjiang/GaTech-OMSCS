############################################
#
# Assignment 1 Supervised Learning
# 
# Author: Wenjun Jiang
#
############################################


# Load Data Sets
## Data Set 1: Housing Data
house.names <- c("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV")
house.data <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"),
                       sep="", header=FALSE, col.names=house.names)
#house.data = read.csv("housing.origin.csv")
#house.names = colnames(house.data)
## Data Set 2: Wholesale Data
wholesale.data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"))
#wholesale.data = read.csv("Wholesale customers data.csv")
wholesale.names <- colnames(wholesale.data)

# 1. Data Exploration
## Step 1.1: Histogram
### 1.1.1: Housing Data
for (i in 1:ncol(house.data)) {
  jpeg(paste0("Histogram of House$", house.names[i], '.jpg'))
  hist(as.numeric(unlist(house.data[i])), col="lightblue", border="pink",
       main=paste0("Histogram of House$", house.names[i]),
       xlab=house.names[i])
  dev.off()
}
### 1.1.2: Wholesale Data
for (i in 1:ncol(wholesale.data)) {
  jpeg(paste0("Histogram of Wholesale$", wholesale.names[i], '.jpg'))
  hist(as.numeric(unlist(wholesale.data[i])), col="lightblue", border="pink",
       main=paste0("Histogram of Wholesale$", wholesale.names[i]),
       xlab=wholesale.names[i])
  dev.off()
}

## Step 1.2: Correlation Plots
library(corrplot)
### 1.2.1: Housing Data
house.corr = cor(house.data)
jpeg("Correlation Plot of Housing.jpg")
corrplot(corr=house.corr, method="ellipse")
dev.off()
### 1.2.2: Wholesale Data
wholesale.corr = cor(wholesale.data)
jpeg("Correlation Plot of Wholesale.jpg")
corrplot(corr=wholesale.corr, method="ellipse")
dev.off()

# Step 1.3: Data Preparation
## 1.3.1: Housing Data
## Categorize MEDV into 5 types: (<10, 1), (10~20, 2), (20~30, 3), (30~40, 4), (>40, 5)
house.data$MEDV = floor(house.data$MEDV/10)+1
replace(house.data$Price.class, house.data$Price.class==6, 5)
colnames(house.data)[house.names=="MEDV"] = "Price.class"
### change to factor type
house.data$Price.class = factor(house.data$Price.class)
## 1.3.2: Wholesale Data
### change to factor type
wholesale.data$Channel = factor(wholesale.data$Channel)

## 1.3.3: Split Training with Testing Data
### load libraries
library(C50)
library(caret)
### set seed
set.seed(2016)
### 1.3.3.1: Housing Data
#### Training Data
house.training.indices = createDataPartition(house.data$Price.class, p=0.8, list=FALSE)
house.training = house.data[house.training.indices,]
#### Testing Data
house.testing = house.data[-house.training.indices,]
### 1.3.3.2: Wholesale Data
#### Training Data
wholesale.training.indices = createDataPartition(wholesale.data$Channel, p=0.8, list=FALSE)
wholesale.training = wholesale.data[wholesale.training.indices,]
#### Testing Data
wholesale.testing = wholesale.data[-wholesale.training.indices,]

# 2. Decision Trees
set.seed(2016)
## 10 folds cross-validation
fitControl = trainControl(method="repeatedcv",number=10,repeats = 10)
## Housing
### Fit
house.decision.tree.fit = train(Price.class~ ., data=house.training, method="C5.0Tree", trControl=fitControl)
### Prediction
house.decision.tree.predict.train = predict(house.decision.tree.fit, newdata=house.training)
house.decision.tree.predict.test = predict(house.decision.tree.fit, newdata=house.testing)
### Report Accuracy & Kappa Values
postResample(house.decision.tree.predict.train, house.training$Price.class)
postResample(house.decision.tree.predict.test,  house.testing$Price.class)
## Wholesale
### Fit
wholesale.decision.tree.fit = train(Channel~ ., data = wholesale.training, 
                                    method="C5.0Tree", trControl=fitControl)
### Prediction
wholesale.decision.tree.predict.train = predict(wholesale.decision.tree.fit, newdata=wholesale.training)
wholesale.decision.tree.predict.test = predict(wholesale.decision.tree.fit, newdata=wholesale.testing)
### Report Accuracy & Kappa Values
postResample(wholesale.decision.tree.predict.train, wholesale.training$Channel)
postResample(wholesale.decision.tree.predict.test,  wholesale.testing$Channel)

#2': Decision Trees with Pruning of the CF value
house.dt.cf.training = rep(0,10)
wholesale.dt.cf.training = rep(0,10)

house.dt.cf.testing = rep(0,10)
wholesale.dt.cf.testing = rep(0,10)
pos = 1
cf.value = seq(0,0.5,length.out = 20)
for(cf in cf.value){
  house.dt.cf.fit = C5.0(Price.class ~ ., data = house.training, trail = 1,
                         control = C5.0Control(noGlobalPruning = FALSE,CF=cf))
  house.tree.predict.insample = predict(house.dt.cf.fit, house.training, type = "class")
  house.tree.predict.outsample = predict(house.dt.cf.fit, house.testing, type = "class")
  in.1 = postResample(house.tree.predict.insample, house.training$Price.class)
  out.1 = postResample(house.tree.predict.outsample, house.testing$Price.class)
  house.dt.cf.training[pos] = as.double(in.1)[1]
  house.dt.cf.testing[pos] = as.double(out.1)[1]
  
  wholesale.dt.cf.fit = C5.0(Channel ~ ., data = wholesale.training, trail = 1,
                             control = C5.0Control(noGlobalPruning = FALSE,CF=cf))
  wholesale.tree.predict.insample = predict(wholesale.dt.cf.fit, wholesale.training, type = "class")
  wholesale.tree.predict.outsample = predict(wholesale.dt.cf.fit, wholesale.testing, type = "class")
  in.2 = postResample(wholesale.tree.predict.insample, wholesale.training$Channel)
  out.2 = postResample(wholesale.tree.predict.outsample, wholesale.testing$Channel)
  wholesale.dt.cf.training[pos] = as.double(in.2)[1]
  wholesale.dt.cf.testing[pos] = as.double(out.2)[1]
  
  pos = pos + 1
}

## Plot
jpeg("Accuracy vs CF for Housing.jpg")
plot(x=cf.value, y=house.dt.cf.training, xlab="Confidence Factor", ylab="accuracy",
     main="Housing Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(house.dt.cf.training, house.dt.cf.testing),max(house.dt.cf.training, house.dt.cf.testing)))
lines(x=cf.value, y=house.dt.cf.testing, col='blue')
legend("right", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()

jpeg("Accuracy vs CF for Wholesale.jpg")
plot(x=cf.value, y=wholesale.dt.cf.training, xlab="Confidence Factor", ylab="accuracy",
     main="Wholesale Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(wholesale.dt.cf.training, wholesale.dt.cf.testing),max(wholesale.dt.cf.training, wholesale.dt.cf.testing)))
lines(x=cf.value, y=wholesale.dt.cf.testing, col='blue')
legend("right", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()

#2'': Decision Trees with Pruning of the minimum case value
house.dt.mc.training = rep(0,10)
wholesale.dt.mc.training = rep(0,10)

house.dt.mc.testing = rep(0,10)
wholesale.dt.mc.testing = rep(0,10)
pos = 1
mc.value = seq(2:11)
for(mc in mc.value){
  house.dt.mc.fit = C5.0(Price.class ~ ., data = house.training, trail = 1,
                         control = C5.0Control(minCases=mc))
  house.tree.predict.insample = predict(house.dt.mc.fit, house.training, type = "class")
  house.tree.predict.outsample = predict(house.dt.mc.fit, house.testing, type = "class")
  in.1 = postResample(house.tree.predict.insample, house.training$Price.class)
  out.1 = postResample(house.tree.predict.outsample, house.testing$Price.class)
  house.dt.mc.training[pos] = as.double(in.1)[1]
  house.dt.mc.testing[pos] = as.double(out.1)[1]
  
  wholesale.dt.mc.fit = C5.0(Channel ~ ., data = wholesale.training, trail = 1,
                            control = C5.0Control(minCases=mc))
  wholesale.tree.predict.insample = predict(wholesale.dt.mc.fit, wholesale.training, type = "class")
  wholesale.tree.predict.outsample = predict(wholesale.dt.mc.fit, wholesale.testing, type = "class")
  in.2 = postResample(wholesale.tree.predict.insample, wholesale.training$Channel)
  out.2 = postResample(wholesale.tree.predict.outsample, wholesale.testing$Channel)
  wholesale.dt.mc.training[pos] = as.double(in.2)[1]
  wholesale.dt.mc.testing[pos] = as.double(out.2)[1]
  
  pos = pos + 1
}

## Plot
jpeg("Accuracy vs MC for Housing.jpg")
plot(x=mc.value, y=house.dt.mc.training, xlab="Minimum Cases", ylab="accuracy",
     main="Housing Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(house.dt.mc.training, house.dt.mc.testing),max(house.dt.mc.training, house.dt.mc.testing)))
lines(x=mc.value, y=house.dt.mc.testing, col='blue')
legend("topright", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()

jpeg("Accuracy vs MC for Wholesale.jpg")
plot(x=mc.value, y=wholesale.dt.mc.training, xlab="Minimum Cases", ylab="accuracy",
     main="Wholesale Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(wholesale.dt.mc.training, wholesale.dt.mc.testing),max(wholesale.dt.mc.training, wholesale.dt.mc.testing)))
lines(x=mc.value, y=wholesale.dt.mc.testing, col='blue')
legend("topright", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()

# 3. Neural Networks
set.seed(2016)
fitControl = trainControl(method = "repeatedcv", number = 10, repeats = 10)
nn.grid = expand.grid( .decay = c(0.1,0.2,0.3,0.5,0.7,0.9), .size = c(2,5,8,10) )
## Housing
house.nn.fit = train(Price.class~ ., data = house.training, method = "nnet",
                     trControl = fitControl, tuneGrid = nn.grid)

house.nn = house.nn.fit$results[1:4]
library(plot3D)
jpeg("Accuracy vs Decay&Size for Housing.jpg")
scatter3D(x=house.nn$decay, y=house.nn$size, z=house.nn$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Decay Rate", ylab="Sizes", zlab="Accuracy",
          main="Neural Networks: Accuracy for Housing")
dev.off()
### Prediction
set.seed(2016)
fitControl = trainControl(method = "none")
house.nn = train(Price.class ~ ., data = house.training, method = "nnet", trControl = fitControl, 
                 verbose = FALSE, tuneGrid = data.frame(decay = 0.7, size = 8))
house.nn.predict.train = predict(house.nn, newdata = house.training)
house.nn.predict.test = predict(house.nn, newdata = house.testing)
#### Report
postResample(house.nn.predict.train, house.training$Price.class)
postResample(house.nn.predict.test,  house.testing$Price.class)

## Wholesale
wholesale.nn.fit = train(Channel~ ., data = wholesale.training, method = "nnet",
                         trControl = fitControl, tuneGrid = nn.grid)
wholesale.nn = wholesale.nn.fit$results[1:4]
jpeg("Accuracy vs Decay&Size for Wholesale.jpg")
scatter3D(x=wholesale.nn$decay, y=wholesale.nn$size, z=wholesale.nn$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Decay Rate", ylab="Sizes", zlab="Accuracy",
          main="Neural Networks: Accuracy for Wholesale")
dev.off()
### Prediction
set.seed(2016)
fitControl = trainControl(method = "none")
wholesale.nn = train(Channel ~ ., data = wholesale.training, method = "nnet", trControl = fitControl, 
                 verbose = FALSE, tuneGrid = data.frame(decay = 0.7, size = 8))
wholesale.nn.predict.train = predict(wholesale.nn, newdata = wholesale.training)
wholesale.nn.predict.test = predict(wholesale.nn, newdata = wholesale.testing)
#### Report
postResample(wholesale.nn.predict.train, wholesale.training$Channel)
postResample(wholesale.nn.predict.test,  wholesale.testing$Channel)

# 4. Boosting
boosting.iterations = c(1,5,10,15,20,30,40,50)
boosting.grid = expand.grid( .winnow = c(TRUE,FALSE), .trials=boosting.iterations, .model="tree" )
## Housing
### Fitting
house.boosting.fit = train(Price.class~ ., data = house.training,
                           method = "C5.0", trControl = fitControl,
                           tuneGrid = boosting.grid)
house.boosting.tree.nowinnow = 
  as.numeric(house.boosting.fit[4]$results[house.boosting.fit[4]$results['winnow']==FALSE][25:32])
house.boosting.tree.winnow = 
  as.numeric(house.boosting.fit[4]$results[house.boosting.fit[4]$results['winnow']==TRUE][25:32])
jpeg("Accuracy vs Winnow for Housing.jpg")
plot(x=boosting.iterations, y=house.boosting.tree.nowinnow, xlab="Boosting Iterations", ylab="Accuracy",
     main="Boosting: Accuracy for Housing", type='l', col='red',
     ylim=c(min(house.boosting.tree.nowinnow, house.boosting.tree.winnow),
            max(house.boosting.tree.nowinnow, house.boosting.tree.winnow)))
lines(x=boosting.iterations, y=house.boosting.tree.winnow, col='blue')
legend("bottomright", legend=c("No Winnow", "Winnow"), col=c("red", "blue"), lty=c(1,1))
dev.off()
### Prediction
set.seed(2016)
fitControl = trainControl(method = "none")
house.boost.fit = train(Price.class ~ ., data=house.training, method="C5.0", trControl=fitControl, 
                        verbose= FALSE, tuneGrid = data.frame(trials=20, model="tree", winnow=FALSE))
house.boost.fit.predict.train = predict(house.boost.fit, newdata = house.training)
house.boost.fit.predict.test = predict(house.boost.fit, newdata = house.testing)
#### report
postResample(house.boost.fit.predict.train, house.training$Price.class)
postResample(house.boost.fit.predict.test,  house.testing$Price.class)

# Wholesale
wholesale.boosting.fit = train(Channel~ ., data = wholesale.training,
                               method = "C5.0", trControl = fitControl,
                               tuneGrid = boosting.grid)
wholesale.boosting.tree.nowinnow = 
  as.numeric(wholesale.boosting.fit[4]$results[wholesale.boosting.fit[4]$results['winnow']==FALSE][25:32])
wholesale.boosting.tree.winnow = 
  as.numeric(wholesale.boosting.fit[4]$results[wholesale.boosting.fit[4]$results['winnow']==TRUE][25:32])
jpeg("Accuracy vs Winnow for Wholesale.jpg")
plot(x=boosting.iterations, y=wholesale.boosting.tree.nowinnow, xlab="Boosting Iterations", ylab="Accuracy",
     main="Boosting: Accuracy for Wholesale", type='l', col='red',
     ylim=c(min(wholesale.boosting.tree.nowinnow, wholesale.boosting.tree.winnow),
            max(wholesale.boosting.tree.nowinnow, wholesale.boosting.tree.winnow)))
lines(x=boosting.iterations, y=wholesale.boosting.tree.winnow, col='blue')
legend("bottomright", legend=c("No Winnow", "Winnow"), col=c("red", "blue"), lty=c(1,1))
dev.off()
### Prediction
set.seed(2016)
fitControl = trainControl(method = "none")
wholesale.boost.fit = train(Channel ~ ., data=wholesale.training, method="C5.0", trControl=fitControl,
                            verbose= FALSE, tuneGrid = data.frame(trials=20, model="tree", winnow=FALSE))
wholesale.boost.fit.predict.train = predict(wholesale.boost.fit, newdata = wholesale.training)
wholesale.boost.fit.predict.test = predict(wholesale.boost.fit, newdata = wholesale.testing)
#### report
postResample(wholesale.boost.fit.predict.train, wholesale.training$Channel)
postResample(wholesale.boost.fit.predict.test,  wholesale.testing$Channel)

# 4. SVM
## 4.1 SVMPoly
set.seed(2016)
fitControl = trainControl(method = "repeatedcv", number = 10, repeats = 10)
svm.poly.grid = expand.grid( .degree = c(1, 2, 3, 4, 5, 6), .C = c(2,5,8,10), .scale = c(0.1) )
## Housing
house.svm.poly.fit = train(Price.class~ ., data = house.training, 
                           method = "svmPoly", trControl = fitControl,
                           tuneGrid = svm.poly.grid )
house.svm.poly = house.svm.poly.fit$results[c(1,2,4,5)]
### Plotting
jpeg("Accuracy vs Degree&C for Housing.jpg")
scatter3D(x=house.svm.poly$degree, y=house.svm.poly$C, z=house.svm.poly$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Degree", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM Poly: Accuracy for Housing")
dev.off()
### predict 
set.seed(2016)
fitControl = trainControl(method = "none")
house.svm.poly = train(Price.class ~ ., data = house.training, method = "svmPoly", trControl = fitControl, 
                       verbose = FALSE, tuneGrid = data.frame(degree = 2, C = 2, scale = 0.1))
house.svm.poly.predict.train = predict(house.svm.poly, newdata = house.training)
house.svm.poly.predict.test = predict(house.svm.poly, newdata = house.testing)
### Report
postResample(house.svm.poly.predict.train, house.training$Price.class)
postResample(house.svm.poly.predict.test,  house.testing$Price.class)

## Wholesale
wholesale.svm.poly.fit = train(Channel~ ., data = wholesale.training, 
                               method = "svmPoly", trControl = fitControl,
                               tuneGrid = svm.poly.grid )
wholesale.svm.poly = wholesale.svm.poly.fit$results[c(1,2,4,5)]
jpeg("Accuracy vs Degree&C for Wholesale.jpg")
scatter3D(x=wholesale.svm.poly$degree, y=wholesale.svm.poly$C, z=wholesale.svm.poly$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Degree", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM Poly: Accuracy for Wholesale")
dev.off()
### predict
set.seed(2016)
fitControl = trainControl(method = "none")
wholesale.svm.poly = train(Price.class ~ ., data = wholesale.training, method = "svmPoly", trControl = fitControl, 
                       verbose = FALSE, tuneGrid = data.frame(degree = 2, C = 2, scale = 0.1))
wholesale.svm.poly.predict.train = predict(wholesale.svm.poly, newdata = wholesale.training)
wholesale.svm.poly.predict.test = predict(wholesale.svm.poly, newdata = wholesale.testing)
### Report
postResample(wholesale.svm.poly.predict.train, wholesale.training$Channel)
postResample(wholesale.svm.poly.predict.test,  wholesale.testing$Channel)

## 4.2 SVM RBF
set.seed(2016)
fitControl = trainControl(method = "repeatedcv", number = 10, repeats = 10)
svm.rbf.grid = expand.grid( .C = c(2,5,8,10), .sigma = c(0.01, 0.1, 0.2, 0.5, 0.7) )
### Housing
#### Fit
house.svm.rbf.fit = train(Price.class~ ., data = house.training, method = "svmRadial",
                          trControl = fitControl, tuneGrid = svm.rbf.grid)
house.svm.rbf = house.svm.rbf.fit$results[1:4]
jpeg("Accuracy vs Sigma&C for Housing.jpg")
scatter3D(x=house.svm.rbf$sigma, y=house.svm.rbf$C, z=house.svm.rbf$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Sigma", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM RBF Kernel: Accuracy for Housing")
dev.off()
#### Predict
set.seed(2016)
fitControl = trainControl(method = "none")
house.svm.rbf = train(Price.class ~ ., data = house.training, method = "svmRadial", trControl = fitControl, 
                      verbose = FALSE, tuneGrid = data.frame( C = 5, sigma = 0.1))
house.svm.rbf.predict.train = predict(house.svm.rbf, newdata = house.training)
house.svm.rbf.predict.test = predict(house.svm.rbf, newdata = house.testing)
##### Report
postResample(house.svm.rbf.predict.train, house.training$Price.class)
postResample(house.svm.rbf.predict.test,  house.testing$Price.class)

### Wholesale
#### Fit
wholesale.svm.rbf.fit = train(Channel ~ ., data = wholesale.training, method = "svmRadial",
                              trControl = fitControl, tuneGrid = svm.rbf.grid)
wholesale.svm.rbf = wholesale.svm.rbf.fit$results[1:4]
jpeg("Accuracy vs Sigma&C for Wholesale.jpg")
scatter3D(x=wholesale.svm.rbf$sigma, y=wholesale.svm.rbf$C, z=wholesale.svm.rbf$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Sigma", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM RBF Kernel: Accuracy for Wholesale")
dev.off()
#### Predict
set.seed(2016)
fitControl = trainControl(method = "none")
wholesale.svm.rbf = train(Channel ~ ., data = wholesale.training, method = "svmRadial", trControl = fitControl, 
                      verbose = FALSE, tuneGrid = data.frame( C = 5, sigma = 0.1))
wholesale.svm.rbf.predict.train = predict(wholesale.svm.rbf, newdata = wholesale.training)
wholesale.svm.rbf.predict.test = predict(wholesale.svm.rbf, newdata = wholesale.testing)
##### Report
postResample(wholesale.svm.rbf.predict.train, wholesale.training$Channel)
postResample(wholesale.svm.rbf.predict.test,  wholesale.testing$Channel)

# 5. KNN
set.seed(2016)
fitControl = trainControl(method = "repeatedcv", number = 10, repeats = 10)
knn.grid = expand.grid( .k=c(1,2,3,4,5,6,7,8,9,10) )
## Housing
house.knn.fit = train(Price.class~ ., data = house.training, 
                      method = "knn", trControl = fitControl,
                      tuneGrid = knn.grid )
house.knn = house.knn.fit$results[1:3]
jpeg("Accuracy vs knn for Housing.jpg")
plot(x=house.knn$k, y=house.knn$Accuracy,
     xlab='# of Neighbors', ylab='Accuracy',
     main='KNN: Accuracy for Housing', col='red', type='l')
dev.off()
## Wholesale
wholesale.knn.fit = train(Channel~ ., data = wholesale.training, 
                          method = "knn", trControl = fitControl,
                          tuneGrid = knn.grid )
wholesale.knn = wholesale.knn.fit$results[1:3]
jpeg("Accuracy vs knn for Wholesale.jpg")
plot(x=wholesale.knn$k, y=wholesale.knn$Accuracy,
     xlab='# of Neighbors', ylab='Accuracy',
     main='KNN: Accuracy for Wholesale', col='red', type='l')
dev.off()
### prediction
set.seed(2016)
fitControl = trainControl(method = "none")
#### Housing
house.knn = train(Price.class ~ ., data = house.training, method = "knn", 
                  trControl = fitControl, tuneGrid = data.frame(k = 5))
house.knn.predict.train = predict(house.knn, newdata = house.training)
house.knn.predict.test = predict(house.knn, newdata = house.testing)
#### Report
postResample(house.knn.predict.train, house.training$Price.class)
postResample(house.knn.predict.test,  house.testing$Price.class)
#### Wholesale
wholesale.knn = train(Channel ~ ., data = wholesale.training, method = "knn", 
                      trControl = fitControl, tuneGrid = data.frame(k = 9))
wholesale.knn.predict.train = predict(wholesale.knn, newdata = wholesale.training)
wholesale.knn.predict.test = predict(wholesale.knn, newdata = wholesale.testing)
#### Report
postResample(wholesale.knn.predict.train, wholesale.training$Channel)
postResample(wholesale.knn.predict.test , wholesale.testing$Channel)

### Learning Curve
## Housing
### Training
house.train.df = data.frame(train.percent = 0.1*c(1:10),
                            tree = c(0.95,0.9383,0.926229508,0.9568,0.9113,0.9262,0.9401,0.9169,0.9317,0.9312),
                            boost=c(1.,1.,0.9508,0.9938,0.9803,0.9959,0.9965,0.9908,1.,0.9975),
                            knn=c(0.825,0.8025,0.8197,0.821,0.8177,0.8156,0.8028,0.7785,0.7568,0.7568),
                            svm=c(1.,1.,0.9672,0.963,0.9754,0.9672,0.9683,0.9415,0.9098,0.9066),
                            nn=c(0.925,0.9136,0.9016,0.8827,0.7291,0.6844,0.8169,0.7631,0.7514,0.7543))
jpeg("Accuracy vs TrainingPerc for Housing TrainingData.jpg")
plot(x=house.train.df$train.percent, y=house.train.df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(0.8*min(house.train.df$tree,house.train.df$boost,house.train.df$knn,house.train.df$svm,house.train.df$nn),
            max(house.train.df$tree,house.train.df$boost,house.train.df$knn,house.train.df$svm,house.train.df$nn)),
     main='Accuracy for Housing: Training', col='red', type='l')
lines(x=house.train.df$train.percent, y=house.train.df$boost, col='blue')
lines(x=house.train.df$train.percent, y=house.train.df$knn, col='green')
lines(x=house.train.df$train.percent, y=house.train.df$svm, col='black')
lines(x=house.train.df$train.percent, y=house.train.df$nn, col='darkorchid')
legend("bottomleft", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()

### Testing
house.test.df = data.frame(train.percent = 0.1*c(1:10),
                           tree = c(0.6162,0.6768,0.6667,0.6768,0.697,0.697,0.7071,0.6768,0.7172,0.7778),
                           boost=c(0.6465,0.6263,0.6364,0.7273,0.7273,0.7273,0.7071,0.7172,0.7475,0.7475),
                           knn=c(0.5556,0.5859,0.6162,0.6061,0.5859,0.596,0.596,0.6162,0.6162,0.6465),
                           svm=c(0.596,0.3939,0.6465,0.7071,0.7172,0.697,0.7273,0.7071,0.7172,0.7172),
                           nn=c(0.5657,0.6768,0.596,0.6667,0.6162,0.5253,0.6364,0.7172,0.7475,0.7374))
jpeg("Accuracy vs TrainingPerc for Housing TestingData.jpg")
plot(x=house.test.df$train.percent, y=house.test.df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(0.8*min(house.test.df$tree,house.test.df$boost,house.test.df$knn,house.test.df$svm,house.test.df$nn),
            max(house.test.df$tree,house.test.df$boost,house.test.df$knn,house.test.df$svm,house.test.df$nn)),
     main='Accuracy for Housing: Testing', col='red', type='l')
lines(x=house.test.df$train.percent, y=house.test.df$boost, col='blue')
lines(x=house.test.df$train.percent, y=house.test.df$knn, col='green')
lines(x=house.test.df$train.percent, y=house.test.df$svm, col='black')
lines(x=house.test.df$train.percent, y=house.test.df$nn, col='darkorchid')
legend("bottomright", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()

## Wholesale
### Training
wholesale.train.df = data.frame(train.percent = 0.1*c(1:10),
                                tree = c(0.9429,0.9571,0.9524,0.9504,0.9659,0.9384,0.9595,0.9504,0.9558,0.9575),
                                boost=c(0.9429,0.9,1.,0.9858,0.9091,0.9431,0.9312,0.9255,0.9338,0.9972),
                                knn=c(0.9429,0.9429,0.9429,0.9504,0.9261,0.9242,0.9352,0.9397,0.9464,0.9348),
                                svm=c(0.4857,0.7857,0.819,0.8794,0.9205,0.9289,0.9271,0.9113,0.9148,0.9065),
                                nn=c(0.9143,0.9,0.9429,0.9291,0.9261,0.9005,0.9352,0.9149,0.8833,0.9065))
jpeg("Accuracy vs TrainingPerc for Wholesale TrainingData.jpg")
plot(x=wholesale.train.df$train.percent, y=wholesale.train.df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(min(wholesale.train.df$tree,wholesale.train.df$boost,wholesale.train.df$knn,wholesale.train.df$svm,wholesale.train.df$nn),
            max(wholesale.train.df$tree,wholesale.train.df$boost,wholesale.train.df$knn,wholesale.train.df$svm,wholesale.train.df$nn)),
     main='Accuracy for Wholesale: Training', col='red', type='l')
lines(x=wholesale.train.df$train.percent, y=wholesale.train.df$boost, col='blue')
lines(x=wholesale.train.df$train.percent, y=wholesale.train.df$knn, col='green')
lines(x=wholesale.train.df$train.percent, y=wholesale.train.df$svm, col='black')
lines(x=wholesale.train.df$train.percent, y=wholesale.train.df$nn, col='darkorchid')
legend("bottomright", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()

### Testing
wholesale.test.df = data.frame(train.percent = 0.1*c(1:10),
                               tree = c(0.8391,0.8621,0.8736,0.9425,0.8966,0.9195,0.908,0.908,0.908,0.8736),
                               boost=c(0.8391,0.9195,0.9195,0.954,0.9425,0.954,0.954,0.954,0.9425,0.9195),
                               knn=c(0.8966,0.8966,0.908,0.8966,0.908,0.908,0.931,0.931,0.931,0.954),
                               svm=c(0.4713,0.8506,0.8391,0.8391,0.9195,0.9425,0.9195,0.931,0.931,0.931),
                               nn=rep(0.9195,10))
jpeg("Accuracy vs TrainingPerc for Wholesale TestingData.jpg")
plot(x=wholesale.test.df$train.percent, y=wholesale.test.df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(min(wholesale.test.df$tree,wholesale.test.df$boost,wholesale.test.df$knn,wholesale.test.df$svm,wholesale.test.df$nn),
            max(wholesale.test.df$tree,wholesale.test.df$boost,wholesale.test.df$knn,wholesale.test.df$svm,wholesale.test.df$nn)),
     main='Accuracy for Wholesale: Testing', col='red', type='l')
lines(x=wholesale.test.df$train.percent, y=wholesale.test.df$boost, col='blue')
lines(x=wholesale.test.df$train.percent, y=wholesale.test.df$knn, col='green')
lines(x=wholesale.test.df$train.percent, y=wholesale.test.df$svm, col='black')
lines(x=wholesale.test.df$train.percent, y=wholesale.test.df$nn, col='darkorchid')
legend("bottomright", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()