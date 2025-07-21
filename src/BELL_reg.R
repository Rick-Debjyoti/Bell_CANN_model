rm(list = ls())
library(LambertW)
library(dplyr) 
library(keras)
library(nloptr)
data = read.csv('/Users/debjyoti_mukherjee/Downloads/freMTPL2freq.csv', header = TRUE, stringsAsFactors = TRUE)
#################################################################################

str(data)
summary(data)

## Min exposure cooresponds to 0.00273 yrs or 1 day 

data$ClaimNb = pmin(data$ClaimNb,4)   
data$Exposure = pmin(data$Exposure,1) 


# Feature pre-processing for GLM as numeric  :

data3 = data
data3$AreaGLM = as.numeric(data3$Area)
data3$VehPowerGLM = as.numeric(pmin(data3$VehPower,9))  
VehAgeGLM = cbind(c(0:110), c(1, rep(2,10), rep(3,100)))
data3$VehAgeGLM = as.numeric(VehAgeGLM[data3$VehAge+1,2])
# data3[,"VehAgeGLM"] =relevel(data3[,"VehAgeGLM"], ref="2")
DrivAgeGLM = cbind(c(18:100), c(rep(1,21-18), rep(2,26-21), rep(3,31-26), rep(4,41-31), rep(5,51-41), rep(6,71-51), rep(7,101-71)))
data3$DrivAgeGLM = as.numeric(DrivAgeGLM[data3$DrivAge-17,2])
# data3[,"DrivAgeGLM"] = relevel(data3[,"DrivAgeGLM"], ref="5")
data3$BonusMalusGLM = as.numeric(pmin(data3$BonusMalus, 150))
data3$DensityGLM = as.numeric(log(data3$Density)) 
data3$RegionGLM = as.numeric(data3$Region)
data3$VehGasGLM = as.numeric(data3$VehGas)
data3$VehBrandGLM = as.numeric(data3$VehBrand)
head(data3)
#################################################################################
## Splitting into training and testing set

set.seed(100)
ll = sample(c(1:nrow(data3)), round(0.9*nrow(data3)), replace = FALSE) 
learn <- data3[ll,]
test <- data3[-ll,]

X_train <- learn[c('VehPowerGLM', 'VehAgeGLM', 'DrivAgeGLM', 'BonusMalusGLM','VehBrandGLM', 'VehGasGLM','DensityGLM', 'RegionGLM','AreaGLM')]
y_train = learn$ClaimNb
Exposure_train = learn$Exposure
n_train = nrow(X_train)

p = ncol(X_train)

X_test <- test[c('VehPowerGLM', 'VehAgeGLM', 'DrivAgeGLM', 'BonusMalusGLM','VehBrandGLM', 'VehGasGLM','DensityGLM', 'RegionGLM','AreaGLM')]
y_test = test$ClaimNb
Exposure_test = test$Exposure
n_test= nrow(X_test)
#################################################################################
## choose training or testing (0 train, 1 test )
a = 0
if(a ==0) {X = X_train;y = y_train;Exposure = Exposure_train;n = n_train} else { X = X_test;y = y_test;Exposure = Exposure_test;n = n_test}


#################################################################################
# testing with a small sample of size 300 
# sample_begin = cbind(Exposure,y,X_train)
# sample= slice_sample(.data = sample_begin, n= 300)
# n = 300
# y = sample$y
# X = subset(sample, select = -c(Exposure ,y))
# Exposure = sample$Exposure
#################################################################################
## Stirling Numbers of the 2-nd kind 
Stirling2 <- function(n,m) 
{ 
  if (0 > m || m > n) stop("'m' must be in 0..n !") 
  k <- 0:m 
  sig <- rep(c(1,-1)*(-1)^m, length= m+1)# 1 for m=0; -1 1 (m=1) 
  ga <- gamma(k+1) 
  round(sum( sig * k^n /(ga * rev(ga)))) 
}
## Bell numbers
fBell <- function(x){
  if (x == 0) return(1)
  vBell <- numeric()
  for(j in 1:x){
    vBell[j] = Stirling2(x,j)
  }
  return(sum(vBell))
}

## Bell pmf 
Bell_pmf = function( mu ,y){
  (exp(1-exp(W(mu)))*((W(mu))^y)*fBell(y))/ factorial(y)
}

ll = function(pars){
  mu = exp(log(Exposure) + as.matrix(X)%*%pars)
  ll = c()
  for (i in 1:n){
    ll[i] = log(Bell_pmf(mu[i], y[i]))
  }
  return(-sum(ll))
}


{t1 <- proc.time()
  opt = optim(par = c(rep(0.001,5),-1, rep(-0.001,3)),fn = ll, method = "BFGS")
  (proc.time()-t1)[3]}
#################################################################################
# Poission glm on same data 

## poi pmf 
POI_pmf = function(lambda , y){
  exp(-lambda)*(lambda^y)/ factorial(y)
}

ll = function(pars){
  lambda = exp(log(Exposure) + as.matrix(X)%*%pars)
  ll = c()
  for (i in 1:n){
    ll[i] = log(POI_pmf(lambda[i], y[i]))
  }
  return(-sum(ll))
}
opt = optim(par =c(rep(0.001,5),-1, rep(-0.001,3)),fn = ll, method = "BFGS")

#################################################################################
# > opt Bell case. (results should improve on dummy-coding) 
# $par
# [1] -0.058907266 -0.825711077 -0.069375417  0.012374439 -0.023627158 -0.036957226 -0.186467433 -0.007628798
# [9]  0.257658974
# 
# $value
# [1] 129394.8
# 
# $counts
# function gradient 
# 112       14 
# pars = c(-0.058907266, -0.825711077, -0.069375417  ,0.012374439, -0.023627158 ,-0.036957226, -0.186467433, -0.007628798,0.257658974)
#################################################################################
# > opt Poisson case  (results should improve on dummy-coding)
# $par
# [1] -0.059662959 -0.820658904 -0.069075686  0.012225486 -0.023556506 -0.042905475 -0.187115136 -0.007633975
# [9]  0.257655488
# 
# $value
# [1] 129830.4
# 
# $counts
# function gradient 
# 116       14 
#################################################################################
## Bell deviance loss:
Bell.Deviance = function(obs, pred)
  {
  loss = rep()
  for (i in 1:length(obs)){
    if (obs[i]==0) 
    {
      loss[i] = 2*(-1+exp(W(pred[i]))) }
      else {
        loss[i] = 2*((exp(W(pred[i])))-(exp(W(obs[i])))+(log((W(obs[i])/W(pred[i]))^(obs[i])))) }
  }
  return(mean(loss))
}     
       # 2*(sum(exp(W(pred)))-sum(exp(W(obs)))+sum(log((W(obs)/W(pred))^(obs))))/length(pred)

## Poisson deviance 
Poisson.Deviance = function(obs,pred){2*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)}
#################################################################################
# predictions and deviances 
#bell
pars = c(-0.058907266, -0.825711077, -0.069375417  ,0.012374439, -0.023627158 ,-0.036957226, -0.186467433, -0.007628798,0.257658974) 
pred_train = exp(log(Exposure_train) + as.matrix(X_train)%*%pars)
pred_test = exp(log(Exposure_test) + as.matrix(X_test)%*%pars)
Bell.Deviance(obs = y_train, pred_train) # in-sample
Bell.Deviance(obs = y_test, pred_test)  # out-sample

#poisson
pars = c(-0.059662959, -0.820658904 ,-0.069075686 , 0.012225486 ,-0.023556506, -0.042905475, -0.187115136 ,-0.007633975, 0.257655488) 
pred_train = exp(log(Exposure_train) + as.matrix(X_train)%*%pars)
pred_test = exp(log(Exposure_test) + as.matrix(X_test)%*%pars)
Poisson.Deviance(obs = y_train, pred_train)  # in-sample
Poisson.Deviance(obs = y_test, pred_test)    # out-sample

#################################################################################
# Bell prediction column addition -> predict for learn and test and combine the two vectors and add as a col to data 
head(data3)

X_data = data3[c('VehPowerGLM', 'VehAgeGLM', 'DrivAgeGLM', 'BonusMalusGLM','VehBrandGLM', 'VehGasGLM','DensityGLM', 'RegionGLM','AreaGLM')]
y_data = data3$ClaimNb
Exposure_data = data3$Exposure
n_data = nrow(X_data)
pars = c(-0.058907266, -0.825711077, -0.069375417  ,0.012374439, -0.023627158 ,-0.036957226, -0.186467433, -0.007628798,0.257658974) 
pred_data= exp(log(Exposure_data) + as.matrix(X_data)%*%pars)
Bell.Deviance(obs = y_data, pred_data) # whole data

data$bellGLM = pred_data[,1]
#################################################################################

## pre-processing and splitting section for ANN, and CANN 

# min-max-scaler:
PreProcess.Continuous = function(var1, data2){
  names(data2)[names(data2) == var1]  = "V1"
  data2$X = as.numeric(data2$V1)
  data2$X = 2*(data2$X-min(data2$X))/(max(data2$X)-min(data2$X))-1
  names(data2)[names(data2) == "V1"]  = var1
  names(data2)[names(data2) == "X"]  <- paste(var1,"X", sep="")
  data2
}

# pre-procecessing function:
Features.PreProcess = function(data2){
  data2 = PreProcess.Continuous("Area", data2)   
  data2 = PreProcess.Continuous("VehPower", data2)   
  data2$VehAge = pmin(data2$VehAge,20)
  data2 = PreProcess.Continuous("VehAge", data2)   
  data2$DrivAge = pmin(data2$DrivAge,90)
  data2 = PreProcess.Continuous("DrivAge", data2)   
  data2$BonusMalus = pmin(data2$BonusMalus,150)
  data2 = PreProcess.Continuous("BonusMalus", data2)   
  data2$VehBrandX = as.integer(data2$VehBrand)-1
  data2$VehGasX <- as.integer(data2$VehGas)-1.5
  data2$Density <- round(log(data2$Density),2)
  data2 <- PreProcess.Continuous("Density", data2)   
  data2$RegionX <- as.integer(data2$Region)-1  # char R11,,R94 to number 0,,21
  data2
}

data2 = Features.PreProcess(data)  # keep original variables and bellGLM (CANN)

set.seed(100)
ll = sample(c(1:nrow(data2)), round(0.9*nrow(data2)), replace = FALSE)
learn = data2[ll,]
test = data2[-ll,]
write.csv(learn, file = "/Users/debjyoti_mukherjee/Downloads/bell_learn.csv")
write.csv(test, file = "/Users/debjyoti_mukherjee/Downloads/bell_test.csv")
#################################################################################(python)
library(rgdal)
# library(rgeos)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(corrplot)

dat <- data %>% 
  mutate(ClaimNb = as.integer(ClaimNb),
         VehAge = pmin(VehAge, 20),
         DrivAge = pmin(DrivAge, 90),
         BonusMalus = round(pmin(BonusMalus, 150) / 10, 0) * 10,
         Density = round(log(Density), 0),
         VehGas = factor(VehGas))


p1 <- ggplot(dat, aes(Exposure)) + geom_histogram()
p2 <- ggplot(dat, aes(x = "Exposure", y = Exposure)) + geom_boxplot() +
  labs(x = "Exposure", y = "frequency", title = "boxplot of exposure")
p3 <- ggplot(dat, aes(ClaimNb)) + geom_histogram() +
  labs(x = "number of claims", y = "frequency", title = "histogram of claims number")
grid.arrange(p1, p2, p3, ncol = 2)


library(ggplot2)
library(dplyr)
library(gridExtra)

# Data manipulation
dat <- data %>%
  mutate(
    ClaimNb = as.integer(ClaimNb),
    VehAge = pmin(VehAge, 20),
    DrivAge = pmin(DrivAge, 90),
    BonusMalus = round(pmin(BonusMalus, 150) / 10, 0) * 10,
    Density = round(log(Density), 0),
    VehGas = factor(VehGas)
  )

# Plot 1: Enhanced histogram of Exposure with colors and bin customization
p1 <- ggplot(dat, aes(x = Exposure)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(x = "Exposure", y = "Frequency", title = "Histogram of Exposure") +
  theme_minimal()

# Plot 2: Violin plot for Exposure with jitter for individual points
p2 <- ggplot(dat, aes(x = "Exposure", y = Exposure)) +
  geom_violin(fill = "lightgreen", alpha = 0.5) +
  geom_jitter(width = 0.2, color = "darkgreen", alpha = 0.7) +
  labs(x = "Exposure", y = "Values", title = "Violin Plot of Exposure") +
  theme_classic()

p3 <- ggplot(dat, aes(ClaimNb)) +
  geom_histogram(binwidth = 1, alpha = 0.7) +
  labs(x = "Number of Claims", y = "Frequency", title = "Histogram of Claims Number") +
  theme_minimal()

# Combine the plots
grid.arrange(p1, p2, p3, ncol = 2)


##

runMultiPlot <- function(dat, VarName) {
  dat <- rename(dat, "VarName" = all_of(VarName))
  out_sum <- dat %>%
    group_by(VarName) %>% 
    summarize(NrObs = length(Exposure),
              Exp = sum(Exposure),
              Nr.Claims = sum(ClaimNb),
              Freq = sum(ClaimNb) / sum(Exposure),
              StDev = sqrt(sum(ClaimNb)) / sum(Exposure))
  # Plot 1
  p1 <- ggplot(out_sum, aes(x = VarName, y = Exp, fill = VarName)) +
    geom_bar(stat = "identity") +
    geom_text(stat = 'identity', aes(label = round(Exp, 0), color = VarName), vjust = -0.5, size = 2.5) +
    labs(x = VarName, y = "Exposure in years", title = "exposure") + theme(legend.position = "none")
  
  # Plot 2
  p2 <- ggplot(out_sum, aes(x = VarName, group = 1)) + geom_point(aes(y = Freq, colour = "observed")) +
    geom_line(aes(y = Freq, colour = "observed"), linetype = "dashed") +
    geom_line(aes(x = as.numeric(VarName), y = pf_freq), color = "red") +
    geom_line(aes(x = as.numeric(VarName), y = Freq + 2 * StDev), color = "red", linetype = "dotted") +
    geom_line(aes(x = as.numeric(VarName), y = Freq - 2 * StDev), color = "red", linetype = "dotted") +
    ylim(0, 0.35) + 
    labs(x = paste(VarName, "groups"), y = "frequency", title = "observed frequency") + theme(legend.position = "none")
  
  # Plot 3
  p3 <- ggplot(out_sum) + geom_bar(stat = "identity", aes(x = VarName, y = Freq, fill = VarName)) +
    geom_line(aes(x = as.numeric(VarName), y = pf_freq), color = "red") + guides(fill = FALSE) +
    labs(x = paste(VarName, "groups"),  y = "frequency", title = "observed frequency") + theme(legend.position = "bottom")
  
  grid.arrange(p1, p2, p3, ncol = 2)
}


p3 <- ggplot(dat, aes(ClaimNb)) +
  geom_histogram(binwidth = 1, alpha = 0.7,fill = "lightgray",color = 'black' ) +
  scale_x_continuous(breaks = seq(0, max(dat$ClaimNb, na.rm = TRUE), by = 1)) + # Exact x values
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) + # Adjust frequency scale
  labs(x = "Number of Claims", y = "Frequency", title = "Histogram of Claims Number") +
  theme_minimal()

p3


p1 <- ggplot(dat, aes(Exposure)) +
  geom_histogram(binwidth = 0.1, fill = "lightgray",color = 'black', alpha = 0.7) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + # Adjust x-axis breaks for Exposure
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) + # Adjust frequency scale
  labs(x = "Exposure", y = "Frequency", title = "Histogram of Exposure") +
  theme_minimal()

p1

p2 <- ggplot(dat, aes(x = "Exposure", y = Exposure)) +
  geom_boxplot(fill = "lightgray", color = "black", alpha = 0.7) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) + # Adjust y-axis scale
  labs(x = "", y = "Exposure", title = "Boxplot of Exposure") +
  theme_minimal()

p2


grid.arrange(p1, p2, p3, ncol= 3)

ggsave("multipanel_plot.png", grid.arrange(p1, p2, p3, ncol= 3), width = 12, height = 4, units = "in", dpi = 300) 




# Load required packages
library(dplyr)
library(ggplot2)
library(tidyr)

# Summarize data
summary_stats <- data3%>%
  summarise(
    Total_Policies = n(),
    Mean_ClaimNb = mean(ClaimNb, na.rm = TRUE),
    Median_ClaimNb = median(ClaimNb, na.rm = TRUE),
    Max_ClaimNb = max(ClaimNb, na.rm = TRUE),
    Mean_Exposure = mean(Exposure, na.rm = TRUE),
    Median_Exposure = median(Exposure, na.rm = TRUE),
    Max_Exposure = max(Exposure, na.rm = TRUE),
    Mean_VehAge = mean(VehAge, na.rm = TRUE),
    Median_VehAge = median(VehAge, na.rm = TRUE),
    Max_VehAge = max(VehAge, na.rm = TRUE),
    Mean_DrivAge = mean(DrivAge, na.rm = TRUE),
    Median_DrivAge = median(DrivAge, na.rm = TRUE),
    Max_DrivAge = max(DrivAge, na.rm = TRUE),
    Mean_BonusMalus = mean(BonusMalus, na.rm = TRUE),
    Median_BonusMalus = median(BonusMalus, na.rm = TRUE),
    Max_BonusMalus = max(BonusMalus, na.rm = TRUE),
    Mean_Density = mean(Density, na.rm = TRUE),
    Median_Density = median(Density, na.rm = TRUE),
    Max_Density = max(Density, na.rm = TRUE)
  )

# Summary for categorical variables
categorical_summary <- df %>%
  select(Area, VehPower, VehBrand, VehGas, Region) %>%
  summarise_all(~ list(table(.))) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Counts")

# Create plots (examples for key variables)
ggplot(df, aes(x = ClaimNb)) +
  geom_histogram(binwidth = 1, fill = "blue", alpha = 0.7) +
  labs(title = "Histogram of ClaimNb", x = "Number of Claims", y = "Frequency")

ggplot(df, aes(x = Area)) +
  geom_bar(fill = "orange", alpha = 0.7) +
  labs(title = "Distribution of Area", x = "Area", y = "Count")

# Save plots as images for inclusion in LaTeX
ggsave("ClaimNb_Histogram.png")
ggsave("Area_Distribution.png")


# Load necessary libraries
library(dplyr)
library(tidyr)

# Select only numeric and integer columns
numeric_cols <- data %>% select(where(is.numeric))

# Compute summary statistics for all numeric and integer columns
summary_stats_all <- numeric_cols %>%
  summarise(across(
    everything(),
    list(
      Mean = ~mean(.x, na.rm = TRUE),
      Median = ~median(.x, na.rm = TRUE),
      Min = ~min(.x, na.rm = TRUE),
      Max = ~max(.x, na.rm = TRUE),
      SD = ~sd(.x, na.rm = TRUE)
    ),
    .names = "{col}_{fn}"
  ))

# Reshape summary_stats_all to a long format for easier readability
summary_stats_long <- summary_stats_all %>%
  pivot_longer(everything(), names_to = c("Variable", ".value"), names_sep = "_")

# Print the summary statistics
print(summary_stats_long)
