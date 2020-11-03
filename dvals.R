env_path = '/home/bilgi/Desktop/lmts'
setwd(env_path)
library(LongMemoryTS)

args = commandArgs(trailingOnly=TRUE)

file_name <- args[1]
print(file_name)

data <- read.csv(file_name)

get_d_values <- function(country) {
  new_country <- country[!is.na(country)]
  
  T <- length(new_country)
  
  result_elw_m <- LongMemoryTS::ELW(data=new_country, m=floor(1+T^0.6),mean.est = "mean")$d
  result_elw_n <- LongMemoryTS::ELW(data=new_country, m=floor(1+T^0.6),mean.est = "none")$d
  
  result_elw2S_v0 <- LongMemoryTS::ELW2S(data=new_country, m=floor(1+T^0.6), trend_order=0, taper = "Velasco")$d
  result_elw2S_v1 <- LongMemoryTS::ELW2S(data=new_country, m=floor(1+T^0.6), trend_order=1, taper = "Velasco")$d
  result_elw2S_h0 <- LongMemoryTS::ELW2S(data=new_country, m=floor(1+T^0.6), trend_order=0, taper = "HC")$d
  result_elw2S_h1 <- LongMemoryTS::ELW2S(data=new_country, m=floor(1+T^0.6), trend_order=1, taper = "HC")$d
  
  result_gph <- LongMemoryTS::gph(X=new_country, m=floor(1+T^0.6), l=1)
    
  result_hou_perron <- LongMemoryTS::Hou.Perron(data=new_country, m=floor(1+T^0.6))$d
  
  result_local_w <- LongMemoryTS::local.W(data=new_country,int = c(-0.5, 2.5), m=floor(1+T^0.6), diff_param=1, taper = c("Velasco"),l=1)$d
  
  result = data.frame(elw_m=result_elw_m, 
                      elw_n=result_elw_n,
                      elw2s_v0=result_elw2S_v0, 
                      elw2s_v1=result_elw2S_v1, 
                      elw2s_h0 =result_elw2S_h0,
                      elw2s_h1 =result_elw2S_h1,
                      gph=result_gph,
                      hou_perron=result_hou_perron,
                      local_w = result_local_w)
  
  return(result)
}


if (file_name =='ln_diff_wb.csv'){
  data = subset(data, select = -c(FRO, LIE, DJI, TCA, SSD))
}


df_list<-lapply(data[2:length(data)],get_d_values)


d_values = do.call(what = rbind, args = df_list)

result_path = paste(env_path,'/d_values.csv',sep="")

write.csv(d_values,result_path)
