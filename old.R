#Data reference
#http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

#load Data
house_data = read.csv("housing_origin.csv")
wholesale_data = read.csv("Wholesale customers data.csv")

#explore the features in data set
#defined a multiplot function
multiplot = function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots = c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout = matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx = as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

#draw the histogram of different features in the housing dataset
library(ggplot2)
p1 = ggplot(house_data, aes(x=CRIM)) + geom_histogram()
p2 = ggplot(house_data, aes(x=ZN)) + geom_histogram()
p3 = ggplot(house_data, aes(x=INDUS)) + geom_histogram()
p4 = ggplot(house_data, aes(x=CHAS)) + geom_histogram()
p5 = ggplot(house_data, aes(x=NOX)) + geom_histogram()
p6 = ggplot(house_data, aes(x=RM)) + geom_histogram()
p7 = ggplot(house_data, aes(x=AGE)) + geom_histogram()
p8 = ggplot(house_data, aes(x=DIS)) + geom_histogram()
p9 = ggplot(house_data, aes(x=RAD)) + geom_histogram()
p10 = ggplot(house_data, aes(x=TAX)) + geom_histogram()
p11 = ggplot(house_data, aes(x=PTRATIO)) + geom_histogram()
p12 = ggplot(house_data, aes(x=B)) + geom_histogram()
p13 = ggplot(house_data, aes(x=LSTAT)) + geom_histogram()
p14 = ggplot(house_data, aes(x=MEDV)) + geom_histogram()

multiplot(p1,p2,p3,p4, cols=2)
multiplot(p5,p6,p7,p8,cols=2)
multiplot(p9,p10,p11,p12,cols=2)
multiplot(p13,p14)

#draw the histogram of different features in the wholesale data set
p1 = ggplot(wholesale_data, aes(x=Channel)) + geom_histogram()
p2 = ggplot(wholesale_data, aes(x=Region)) + geom_histogram()
p3 = ggplot(wholesale_data, aes(x=Fresh)) + geom_histogram()
p4 = ggplot(wholesale_data, aes(x=Milk)) + geom_histogram()
p5 = ggplot(wholesale_data, aes(x=Grocery)) + geom_histogram()
p6 = ggplot(wholesale_data, aes(x=Frozen)) + geom_histogram()
p7 = ggplot(wholesale_data, aes(x=Detergents_Paper)) + geom_histogram()
p8 = ggplot(wholesale_data, aes(x=Delicassen)) + geom_histogram()
multiplot(p1,p2,p3,p4, cols=2)
multiplot(p5,p6,p7,p8,cols=2)

#demonstrate the correlationships among the attributes.
library(corrplot)
house_corr = cor(house_data) 
wholesale_corr = cor(wholesale_data)

corrplot(house_corr,tl.pos='n',method = "color")
corrplot(wholesale_corr, tl.pos='n',method = "color")

library(C50)
library(caret)

# 10 folds cross-validation
fitControl = trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

## split the data into train data and test data
house_data = read.csv("housing.csv")
# change the datatype of target variable into factor type
house_data$Price_class = factor(house_data$Price_class )

wholesale_data = read.csv("Wholesale customers data.csv")
wholesale_data$Channel = factor(wholesale_data$Channel)

### split the data
set.seed(2016)
house_in_train = createDataPartition(house_data$Price_class, p = .8, list = FALSE)
house_train = house_data[house_in_train,]
house_test = house_data[-house_in_train,]

set.seed(2016)
wholesale_in_train = createDataPartition(wholesale_data$Channel, p = .8, list = FALSE)
wholesale_train = wholesale_data[wholesale_in_train,]
wholesale_test = wholesale_data[-wholesale_in_train,]


##cross validate a single tree
set.seed(2016)
house_tree_fit = train(Price_class~ ., data = house_train, 
                 method = "C5.0Tree", 
                 trControl = fitControl )

wholesale_tree_fit = train(Channel~ ., data = wholesale_train, 
                   method = "C5.0Tree", 
                   trControl = fitControl )

##predict
##predict single tree
fitControl = trainControl(method = "none")
house_train_tree = rep(0,10)
house_test_tree = rep(0,10)
wholesale_train_tree =rep(0,10)
wholesale_test_tree =rep(0,10)
tree_time = rep(0,10)
for (i in 1:10){
  start = Sys.time()
  len_house = nrow(house_train)
  len_wholsale = nrow(wholesale_train)
  len_house_train = floor(0.1*i*len_house)
  len_whholesale_train = floor(0.1*i*len_wholsale)
  house_train_part = house_train[1:len_house_train,]
  wholesale_train_part = wholesale_train[1:len_whholesale_train,]
  house_train_part$Price_class = factor(house_train_part$Price_class)
  set.seed(2016)
  house_tree_fit = train(Price_class~ ., data = house_train_part,
                          method = "C5.0Tree",
                          trControl = fitControl )
  wholesale_tree_fit = train(Channel~ ., data = wholesale_train_part,
                              method = "C5.0Tree",
                              trControl = fitControl )
  ##predict
  house_tree_predict_train = predict(house_tree_fit, newdata = house_train_part)
  house_tree_predict_test = predict(house_tree_fit, newdata = house_test)
  wholesale_tree_predict_train = predict(wholesale_tree_fit, newdata = wholesale_train_part)
  wholesale_tree_predict_test = predict(wholesale_tree_fit, newdata = wholesale_test)
  house_train_tree[i] = as.double(postResample(house_tree_predict_train, house_train_part$Price_class))[1]
  wholesale_train_tree[i]= as.double(postResample(wholesale_tree_predict_train, wholesale_train_part$Channel))[1]
  house_test_tree[i] = as.double(postResample(house_tree_predict_test, house_test$Price_class))[1]
  wholesale_test_tree[i]= as.double(postResample(wholesale_tree_predict_test, wholesale_test$Channel))[1]
  end = Sys.time()
  tree_time[i] = end - start
}


#Decision Tree
###Adjust the CF value
sample_in_1_acc = rep(0,10)
sample_in_2_acc = rep(0,10)

sample_out_1_acc = rep(0,10)
sample_out_2_acc = rep(0,10)
pos = 1
cf_value = seq(0,0.5,length.out = 20)
for(i in cf_value){
  house_tree_prune_fit = C5.0(Price_class ~ ., data = house_train, trail = 1,
                           control = C5.0Control(noGlobalPruning = FALSE,CF=i))
  house_tree_predict_insample = predict(house_tree_prune_fit, house_train, type = "class")
  house_tree_predict_outsample = predict(house_tree_prune_fit, house_test, type = "class")
  in_1 = postResample(house_tree_predict_insample, house_train$Price_class)
  out_1 = postResample(house_tree_predict_outsample, house_test$Price_class)
  sample_in_1_acc[pos] = as.double(in_1)[1]
  sample_out_1_acc[pos] = as.double(out_1)[1]
  
  
  wholesale_tree_prune_fit =C5.0(Channel ~ ., data = wholesale_train, trail = 1,
                              control = C5.0Control(noGlobalPruning = FALSE,CF=i))
  wholesale_tree_predict_insample = predict(wholesale_tree_prune_fit, wholesale_train, type = "class")
  wholesale_tree_predict_outsample = predict(wholesale_tree_prune_fit, wholesale_test,  type = "class")
  in_2 = postResample(wholesale_tree_predict_insample, wholesale_train$Channel)
  out_2 = postResample(wholesale_tree_predict_outsample, wholesale_test$Channel)
  sample_in_2_acc[pos] = as.double(in_2)[1]
  sample_out_2_acc[pos] = as.double(out_2)[1]

  pos = pos + 1
  print(pos)
}

## Plot
jpeg("Accuracy vs CF for Housing.jpg")
plot(x=cf_value, y=sample_in_1_acc, xlab="Confidence Factor", ylab="accuracy",
     main="Housing Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(sample_in_1_acc, sample_out_1_acc),max(sample_in_1_acc, sample_out_1_acc)))
lines(x=cf_value, y=sample_out_1_acc, col='blue')
legend("right", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()

jpeg("Accuracy vs CF for Wholesale.jpg")
plot(x=cf_value, y=sample_in_2_acc, xlab="Confidence Factor", ylab="accuracy",
     main="Wholesale Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(sample_in_2_acc, sample_out_2_acc),max(sample_in_2_acc, sample_out_2_acc)))
lines(x=cf_value, y=sample_out_2_acc, col='blue')
legend("right", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()


###adjust min case
sample_in_1_acc = rep(0,10)
sample_in_2_acc = rep(0,10)

sample_out_1_acc = rep(0,10)
sample_out_2_acc = rep(0,10)
pos = 1
min_case = seq(2:11)
for(i in min_case){
  house_tree_prune_fit = C5.0(Price_class ~ ., data = house_train, trail = 1,
                               control = C5.0Control(minCases = i))
  house_tree_predict_insample = predict(house_tree_prune_fit, house_train, type = "class")
  house_tree_predict_outsample = predict(house_tree_prune_fit, house_test, type = "class")
  in_1 = postResample(house_tree_predict_insample, house_train$Price_class)
  out_1 = postResample(house_tree_predict_outsample, house_test$Price_class)
  sample_in_1_acc[pos] = as.double(in_1)[1]
  sample_out_1_acc[pos] = as.double(out_1)[1]
  
  
  wholesale_tree_prune_fit =C5.0(Channel ~ ., data = wholesale_train, trail = 1,
                                  control = C5.0Control(minCases = i))
  wholesale_tree_predict_insample = predict(wholesale_tree_prune_fit, wholesale_train, type = "class")
  wholesale_tree_predict_outsample = predict(wholesale_tree_prune_fit, wholesale_test,  type = "class")
  in_2 = postResample(wholesale_tree_predict_insample, wholesale_train$Channel)
  out_2 = postResample(wholesale_tree_predict_outsample, wholesale_test$Channel)
  sample_in_2_acc[pos] = as.double(in_2)[1]
  sample_out_2_acc[pos] = as.double(out_2)[1]
  pos = pos + 1
  print(pos)
}

## Plot
jpeg("Accuracy vs MC for Housing.jpg")
plot(x=min_case, y=sample_in_1_acc, xlab="Minimum Cases", ylab="accuracy",
     main="Housing Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(sample_in_1_acc, sample_out_1_acc),max(sample_in_1_acc, sample_out_1_acc)))
lines(x=min_case, y=sample_out_1_acc, col='blue')
legend("topright", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()

jpeg("Accuracy vs MC for Wholesale.jpg")
plot(x=min_case, y=sample_in_2_acc, xlab="Minimum Cases", ylab="accuracy",
     main="Wholesale Data: Training and Testing Accuracy", type='l', col='red',
     ylim=c(min(sample_in_2_acc, sample_out_2_acc),max(sample_in_2_acc, sample_out_2_acc)))
lines(x=min_case, y=sample_out_2_acc, col='blue')
legend("topright", legend=c("Training", "Testing"), col=c("red", "blue"), lty=c(1,1))
dev.off()


### Boosting Method

##cross validate 
set.seed(2016)
fitControl = trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)


boosting_iterations = c(1,5,10,15,20,30,40,50)
boost_grid = expand.grid( .winnow = c(TRUE,FALSE), .trials=boosting_iterations, .model="tree" )
# Housing
house_tree_fit = train(Price_class~ ., data = house_train, 
                        method = "C5.0", trControl = fitControl,
                        tuneGrid = boost_grid)
house_boosting_tree_nowinnow = 
  as.numeric(house_tree_fit[4]$results[house_tree_fit[4]$results['winnow']==FALSE][25:32])
house_boosting_tree_winnow = 
  as.numeric(house_tree_fit[4]$results[house_tree_fit[4]$results['winnow']==TRUE][25:32])
jpeg("Accuracy vs Winnow for Housing.jpg")
plot(x=boosting_iterations, y=house_boosting_tree_nowinnow, xlab="Boosting Iterations", ylab="Accuracy",
     main="Boosting: Accuracy for Housing", type='l', col='red',
     ylim=c(min(house_boosting_tree_nowinnow, house_boosting_tree_winnow),
            max(house_boosting_tree_nowinnow, house_boosting_tree_winnow)))
lines(x=boosting_iterations, y=house_boosting_tree_winnow, col='blue')
legend("bottomright", legend=c("No Winnow", "Winnow"), col=c("red", "blue"), lty=c(1,1))
dev.off()

# Wholesale
wholesale_tree_fit = train(Channel~ ., data = wholesale_train, 
                            method = "C5.0", trControl = fitControl,
                            tuneGrid = boost_grid)
wholesale_boosting_tree_nowinnow = 
  as.numeric(wholesale_tree_fit[4]$results[wholesale_tree_fit[4]$results['winnow']==FALSE][25:32])
wholesale_boosting_tree_winnow = 
  as.numeric(wholesale_tree_fit[4]$results[wholesale_tree_fit[4]$results['winnow']==TRUE][25:32])
jpeg("Accuracy vs Winnow for Wholesale.jpg")
plot(x=boosting_iterations, y=wholesale_boosting_tree_nowinnow, xlab="Boosting Iterations", ylab="Accuracy",
     main="Boosting: Accuracy for Wholesale", type='l', col='red',
     ylim=c(min(wholesale_boosting_tree_nowinnow, wholesale_boosting_tree_winnow),
            max(wholesale_boosting_tree_nowinnow, wholesale_boosting_tree_winnow)))
lines(x=boosting_iterations, y=wholesale_boosting_tree_winnow, col='blue')
legend("bottomright", legend=c("No Winnow", "Winnow"), col=c("red", "blue"), lty=c(1,1))
dev.off()


###select the model
##predict
#house_boost_tree_fit = C5.0(Price_class~., data = house_train, trail = 10,winnow=FALSE)
set.seed(2016)
fitControl = trainControl(method = "none")
house_boost_tree = train(Price_class ~ ., data = house_train, 
                          method = "C5.0", 
                          trControl = fitControl, 
                          verbose = FALSE, 
                 ## Only a single model can be passed to the
                 ## function when no resampling is used:
                          tuneGrid = data.frame(trials = 20, model = "tree", winnow = FALSE))


house_boost_tree_predict_train = predict(house_boost_tree, newdata = house_train)
house_boost_tree_predict_test = predict(house_boost_tree, newdata = house_test)

postResample(house_boost_tree_predict_train, house_train$Price_class)
postResample(house_boost_tree_predict_test,  house_test$Price_class)


wholesale_boost_tree = train(Channel ~ ., data = wholesale_train, 
                          method = "C5.0", 
                          trControl = fitControl, 
                          verbose = FALSE, 
                          ## Only a single model can be passed to the
                          ## function when no resampling is used:
                          tuneGrid = data.frame(trials = 40, model = "tree", winnow = FALSE))

wholesale_boost_tree_predict_train = predict(wholesale_boost_tree, newdata = wholesale_train)
wholesale_boost_tree_predict_test = predict(wholesale_boost_tree, newdata = wholesale_test)

postResample(wholesale_boost_tree_predict_train, wholesale_train$Channel)
postResample(wholesale_boost_tree_predict_test,  wholesale_test$Channel)


###Nerual Networks
##cross validate 
set.seed(2016)
fitControl = trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

decay_factor = seq(0.1, 1.0, by=0.1)
nn_size = seq(1, 15, by=2)
nn_grid = expand.grid( .decay = decay_factor, .size = nn_size)

house_nn_fit = train(Price_class~ ., data = house_train, 
                        method = "nnet", trControl = fitControl,
                        tuneGrid = nn_grid)


wholesale_nn_fit = train(Channel~ ., data = wholesale_train, 
                            method = "nnet", trControl = fitControl,
                            tuneGrid = nn_grid)
plot(house_nn_fit)
plot(wholesale_nn_fit)

###parameter tuning further
nn_grid_2 = expand.grid( .decay = c(0.1,0.2,0.3,0.5,0.7,0.9), .size = c(2,5,8,10) )
## Housing
house_nn_fit_2 = train(Price_class~ ., data = house_train, 
                      method = "nnet", trControl = fitControl,
                      tuneGrid = nn_grid_2)

house_nn = house_nn_fit_2$results[1:4]
library(plot3D)
jpeg("Accuracy vs Decay&Size for Housing.jpg")
scatter3D(x=house_nn$decay, y=house_nn$size, z=house_nn$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Decay Rate", ylab="Sizes", zlab="Accuracy",
          main="Neural Networks: Accuracy for Housing")
dev.off()
## Wholesale
wholesale_nn_fit_2 = train(Channel~ ., data = wholesale_train, 
                          method = "nnet", trControl = fitControl,
                          tuneGrid = nn_grid_2)
wholesale_nn = wholesale_nn_fit_2$results[1:4]
jpeg("Accuracy vs Decay&Size for Wholesale.jpg")
scatter3D(x=wholesale_nn$decay, y=wholesale_nn$size, z=wholesale_nn$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Decay Rate", ylab="Sizes", zlab="Accuracy",
          main="Neural Networks: Accuracy for Wholesale")
dev.off()


###select the model 
###predict 
set.seed(2016)
fitControl = trainControl(method = "none")
house_nn = train(Price_class ~ ., data = house_train, 
                          method = "nnet", 
                          trControl = fitControl, 
                          verbose = FALSE, 
                          ## Only a single model can be passed to the
                          ## function when no resampling is used:
                          tuneGrid = data.frame(decay = 0.7, size = 8))

house_nn_predict_train = predict(house_nn, newdata = house_train)
house_nn_predict_test = predict(house_nn, newdata = house_test)

postResample(house_nn_predict_train, house_train$Price_class)
postResample(house_nn_predict_test,  house_test$Price_class)

wholesale_nn = train(Channel ~ ., data = wholesale_train, 
                              method = "nnet", 
                              trControl = fitControl, 
                              verbose = FALSE, 
                              ## Only a single model can be passed to the
                              ## function when no resampling is used: 
                              tuneGrid = data.frame(decay = 0.7, size = 2))

wholesale_nn_predict_train = predict(wholesale_nn, newdata = wholesale_train)
wholesale_nn_predict_test = predict(wholesale_nn, newdata = wholesale_test)

postResample(wholesale_nn_predict_train, wholesale_train$Channel)
postResample(wholesale_boost_tree_predict_test,  wholesale_test$Channel)



###SVM parmeter tuning
## SVMpoly
set.seed(2016)
fitControl = trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

svm_poly_grid = expand.grid( .degree = c(1, 2, 3, 4, 5, 6), .C = c(2,5,8,10), .scale = c(0.1) )
## Housing
house_svm_poly_fit = train(Price_class~ ., data = house_train, 
                        method = "svmPoly", trControl = fitControl,
                        tuneGrid = svm_poly_grid )

house_svm_poly = house_svm_poly_fit$results[c(1,2,4,5)]
jpeg("Accuracy vs Degree&C for Housing.jpg")
scatter3D(x=house_svm_poly$degree, y=house_svm_poly$C, z=house_svm_poly$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Degree", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM Poly: Accuracy for Housing")
dev.off()
## Wholesale
wholesale_svm_poly_fit = train(Channel~ ., data = wholesale_train, 
                            method = "svmPoly", trControl = fitControl,
                            tuneGrid = svm_poly_grid )
wholesale_svm_poly = wholesale_svm_poly_fit$results[c(1,2,4,5)]
jpeg("Accuracy vs Degree&C for Wholesale.jpg")
scatter3D(x=wholesale_svm_poly$degree, y=wholesale_svm_poly$C, z=wholesale_svm_poly$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Degree", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM Poly: Accuracy for Wholesale")
dev.off()


##select the model
##predict 
set.seed(2016)
fitControl = trainControl(method = "none")
house_svm_poly = train(Price_class ~ ., data = house_train, 
                  method = "svmPoly", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  ## Only a single model can be passed to the
                  ## function when no resampling is used:
                  tuneGrid = data.frame(degree = 2, C = 2, scale = 0.1))

house_svm_poly_predict_train = predict(house_svm_poly, newdata = house_train)
house_svm_poly_predict_test = predict(house_svm_poly, newdata = house_test)

postResample(house_svm_poly_predict_train, house_train$Price_class)
postResample(house_svm_poly_predict_test,  house_test$Price_class)


wholesale_svm_poly = train(Channel ~ ., data = wholesale_train, 
                      method = "svmPoly", 
                      trControl = fitControl, 
                      verbose = FALSE, 
                      ## Only a single model can be passed to the
                      ## function when no resampling is used: 
                      tuneGrid = data.frame(degree = 1, C = 5, scale = 0.1))

wholesale_svm_poly_predict_train = predict(wholesale_svm_poly, newdata = wholesale_train)
wholesale_svm_poly_predict_test = predict(wholesale_svm_poly, newdata = wholesale_test)

postResample(wholesale_svm_poly_predict_train, wholesale_train$Channel)
postResample(wholesale_svm_poly_predict_test , wholesale_test$Channel)


## SVM RBF 
set.seed(2016)
fitControl = trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

svm_rbf_grid = expand.grid( .C = c(2,5,8,10), .sigma = c(0.01, 0.1, 0.2, 0.5, 0.7) )

house_svm_rbf_fit = train(Price_class~ ., data = house_train, 
                            method = "svmRadial", trControl = fitControl,
                            tuneGrid = svm_rbf_grid )
house_svm_rbf = house_svm_rbf_fit$results[1:4]
jpeg("Accuracy vs Sigma&C for Housing.jpg")
scatter3D(x=house_svm_rbf$sigma, y=house_svm_rbf$C, z=house_svm_rbf$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Sigma", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM RBF Kernel: Accuracy for Housing")
dev.off()
## Wholesale
wholesale_svm_rbf_fit = train(Channel~ ., data = wholesale_train, 
                                method = "svmRadial", trControl = fitControl,
                                tuneGrid = svm_rbf_grid )
wholesale_svm_rbf = wholesale_svm_rbf_fit$results[1:4]
jpeg("Accuracy vs Sigma&C for Wholesale.jpg")
scatter3D(x=wholesale_svm_rbf$sigma, y=wholesale_svm_rbf$C, z=wholesale_svm_rbf$Accuracy,
          bty='g', pch=20, cex=2, ticktype='detailed',
          xlab="Sigma", ylab="Cost of Constraints Violation", zlab="Accuracy",
          main="SVM RBF Kernel: Accuracy for Wholesale")
dev.off()

###predict
set.seed(2016)
fitControl = trainControl(method = "none")
house_svm_rbf = train(Price_class ~ ., data = house_train, 
                        method = "svmRadial", 
                        trControl = fitControl, 
                        verbose = FALSE, 
                        ## Only a single model can be passed to the
                        ## function when no resampling is used:
                        tuneGrid = data.frame( C = 5, sigma = 0.1))

house_svm_rbf_predict_train = predict(house_svm_rbf, newdata = house_train)
house_svm_rbf_predict_test = predict(house_svm_rbf, newdata = house_test)

postResample(house_svm_rbf_predict_train, house_train$Price_class)
postResample(house_svm_rbf_predict_test,  house_test$Price_class)


wholesale_svm_rbf = train(Channel ~ ., data = wholesale_train, 
                            method = "svmRadial", 
                            trControl = fitControl, 
                            verbose = FALSE, 
                            ## Only a single model can be passed to the
                            ## function when no resampling is used: 
                            tuneGrid = data.frame(C = 10, sigma = 0.01))

wholesale_svm_rbf_predict_train = predict(wholesale_svm_rbf, newdata = wholesale_train)
wholesale_svm_rbf_predict_test = predict(wholesale_svm_rbf, newdata = wholesale_test)

postResample(wholesale_svm_rbf_predict_train, wholesale_train$Channel)
postResample(wholesale_svm_rbf_predict_test , wholesale_test$Channel)


###KNN
set.seed(2016)
fitControl = trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

knn_grid = expand.grid( .k=c(1,2,3,4,5,6,7,8,9,10) )
## Housing
house_knn_fit = train(Price_class~ ., data = house_train, 
                           method = "knn", trControl = fitControl,
                           tuneGrid = knn_grid )
house_knn = house_knn_fit$results[1:3]
jpeg("Accuracy vs knn for Housing.jpg")
plot(x=house_knn$k, y=house_knn$Accuracy,
     xlab='# of Neighbors', ylab='Accuracy',
     main='KNN: Accuracy for Housing', col='red', type='l')
dev.off()
## Wholesale
wholesale_knn_fit = train(Channel~ ., data = wholesale_train, 
                               method = "knn", trControl = fitControl,
                               tuneGrid = knn_grid )
wholesale_knn = wholesale_knn_fit$results[1:3]
jpeg("Accuracy vs knn for Wholesale.jpg")
plot(x=wholesale_knn$k, y=wholesale_knn$Accuracy,
     xlab='# of Neighbors', ylab='Accuracy',
     main='KNN: Accuracy for Wholesale', col='red', type='l')
dev.off()


###KNN
### prediction
set.seed(2016)
fitControl = trainControl(method = "none")
house_knn = train(Price_class ~ ., data = house_train, 
                       method = "knn", 
                       trControl = fitControl, 
                       ## Only a single model can be passed to the
                       ## function when no resampling is used:
                       tuneGrid = data.frame(k = 5))

house_knn_predict_train = predict(house_knn, newdata = house_train)
house_knn_predict_test = predict(house_knn, newdata = house_test)

postResample(house_knn_predict_train, house_train$Price_class)
postResample(house_knn_predict_test,  house_test$Price_class)


wholesale_knn = train(Channel ~ ., data = wholesale_train, 
                           method = "knn", 
                           trControl = fitControl, 
                           ## Only a single model can be passed to the
                           ## function when no resampling is used: 
                           tuneGrid = data.frame(k = 9))

wholesale_knn_predict_train = predict(wholesale_knn, newdata = wholesale_train)
wholesale_knn_predict_test = predict(wholesale_knn, newdata = wholesale_test)

postResample(wholesale_knn_predict_train, wholesale_train$Channel)
postResample(wholesale_knn_predict_test , wholesale_test$Channel)


### Learning Curve
## Housing
### Training
house_train_df = data.frame(train_percent = 0.1*c(1:10),
                            tree = c(0.95,0.9383,0.926229508,0.9568,0.9113,0.9262,0.9401,0.9169,0.9317,0.9312),
                            boost=c(1.,1.,0.9508,0.9938,0.9803,0.9959,0.9965,0.9908,1.,0.9975),
                            knn=c(0.825,0.8025,0.8197,0.821,0.8177,0.8156,0.8028,0.7785,0.7568,0.7568),
                            svm=c(1.,1.,0.9672,0.963,0.9754,0.9672,0.9683,0.9415,0.9098,0.9066),
                            nn=c(0.925,0.9136,0.9016,0.8827,0.7291,0.6844,0.8169,0.7631,0.7514,0.7543))
jpeg("Accuracy vs TrainingPerc for Housing TrainingData.jpg")
plot(x=house_train_df$train_percent, y=house_train_df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(0.8*min(house_train_df$tree,house_train_df$boost,house_train_df$knn,house_train_df$svm,house_train_df$nn),
            max(house_train_df$tree,house_train_df$boost,house_train_df$knn,house_train_df$svm,house_train_df$nn)),
     main='Accuracy for Housing: Training', col='red', type='l')
lines(x=house_train_df$train_percent, y=house_train_df$boost, col='blue')
lines(x=house_train_df$train_percent, y=house_train_df$knn, col='green')
lines(x=house_train_df$train_percent, y=house_train_df$svm, col='black')
lines(x=house_train_df$train_percent, y=house_train_df$nn, col='darkorchid')
legend("bottomleft", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()

### Testing
house_test_df = data.frame(train_percent = 0.1*c(1:10),
                           tree = c(0.6162,0.6768,0.6667,0.6768,0.697,0.697,0.7071,0.6768,0.7172,0.7778),
                           boost=c(0.6465,0.6263,0.6364,0.7273,0.7273,0.7273,0.7071,0.7172,0.7475,0.7475),
                           knn=c(0.5556,0.5859,0.6162,0.6061,0.5859,0.596,0.596,0.6162,0.6162,0.6465),
                           svm=c(0.596,0.3939,0.6465,0.7071,0.7172,0.697,0.7273,0.7071,0.7172,0.7172),
                           nn=c(0.5657,0.6768,0.596,0.6667,0.6162,0.5253,0.6364,0.7172,0.7475,0.7374))
jpeg("Accuracy vs TrainingPerc for Housing TestingData.jpg")
plot(x=house_test_df$train_percent, y=house_test_df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(0.8*min(house_test_df$tree,house_test_df$boost,house_test_df$knn,house_test_df$svm,house_test_df$nn),
            max(house_test_df$tree,house_test_df$boost,house_test_df$knn,house_test_df$svm,house_test_df$nn)),
     main='Accuracy for Housing: Testing', col='red', type='l')
lines(x=house_test_df$train_percent, y=house_test_df$boost, col='blue')
lines(x=house_test_df$train_percent, y=house_test_df$knn, col='green')
lines(x=house_test_df$train_percent, y=house_test_df$svm, col='black')
lines(x=house_test_df$train_percent, y=house_test_df$nn, col='darkorchid')
legend("bottomright", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()

## Wholesale
### Training
wholesale_train_df = data.frame(train_percent = 0.1*c(1:10),
                                tree = c(0.9429,0.9571,0.9524,0.9504,0.9659,0.9384,0.9595,0.9504,0.9558,0.9575),
                                boost=c(0.9429,0.9,1.,0.9858,0.9091,0.9431,0.9312,0.9255,0.9338,0.9972),
                                knn=c(0.9429,0.9429,0.9429,0.9504,0.9261,0.9242,0.9352,0.9397,0.9464,0.9348),
                                svm=c(0.4857,0.7857,0.819,0.8794,0.9205,0.9289,0.9271,0.9113,0.9148,0.9065),
                                nn=c(0.9143,0.9,0.9429,0.9291,0.9261,0.9005,0.9352,0.9149,0.8833,0.9065))
jpeg("Accuracy vs TrainingPerc for Wholesale TrainingData.jpg")
plot(x=wholesale_train_df$train_percent, y=wholesale_train_df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(min(wholesale_train_df$tree,wholesale_train_df$boost,wholesale_train_df$knn,wholesale_train_df$svm,wholesale_train_df$nn),
            max(wholesale_train_df$tree,wholesale_train_df$boost,wholesale_train_df$knn,wholesale_train_df$svm,wholesale_train_df$nn)),
     main='Accuracy for Wholesale: Training', col='red', type='l')
lines(x=wholesale_train_df$train_percent, y=wholesale_train_df$boost, col='blue')
lines(x=wholesale_train_df$train_percent, y=wholesale_train_df$knn, col='green')
lines(x=wholesale_train_df$train_percent, y=wholesale_train_df$svm, col='black')
lines(x=wholesale_train_df$train_percent, y=wholesale_train_df$nn, col='darkorchid')
legend("bottomright", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()

### Testing
wholesale_test_df = data.frame(train_percent = 0.1*c(1:10),
                               tree = c(0.8391,0.8621,0.8736,0.9425,0.8966,0.9195,0.908,0.908,0.908,0.8736),
                               boost=c(0.8391,0.9195,0.9195,0.954,0.9425,0.954,0.954,0.954,0.9425,0.9195),
                               knn=c(0.8966,0.8966,0.908,0.8966,0.908,0.908,0.931,0.931,0.931,0.954),
                               svm=c(0.4713,0.8506,0.8391,0.8391,0.9195,0.9425,0.9195,0.931,0.931,0.931),
                               nn=rep(0.9195,10))
jpeg("Accuracy vs TrainingPerc for Wholesale TestingData.jpg")
plot(x=wholesale_test_df$train_percent, y=wholesale_test_df$tree,
     xlab='% of Training Data', ylab='Accuracy',
     ylim=c(min(wholesale_test_df$tree,wholesale_test_df$boost,wholesale_test_df$knn,wholesale_test_df$svm,wholesale_test_df$nn),
            max(wholesale_test_df$tree,wholesale_test_df$boost,wholesale_test_df$knn,wholesale_test_df$svm,wholesale_test_df$nn)),
     main='Accuracy for Wholesale: Testing', col='red', type='l')
lines(x=wholesale_test_df$train_percent, y=wholesale_test_df$boost, col='blue')
lines(x=wholesale_test_df$train_percent, y=wholesale_test_df$knn, col='green')
lines(x=wholesale_test_df$train_percent, y=wholesale_test_df$svm, col='black')
lines(x=wholesale_test_df$train_percent, y=wholesale_test_df$nn, col='darkorchid')
legend("bottomright", legend=c("Decision Trees","Boosting","KNN","SVM","Neural Networks"),
       col=c("red","blue","green","black","darkorchid"), lty=seq(1,5))
dev.off()