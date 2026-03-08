#Etude préliminaire des données DataBattle IAPau 2026

library(timeDate)
library(dplyr)
library(lubridate)
df <- read.csv2("segment_alerts_all_airports_train.csv", sep=",", header=TRUE)

select(df, ends_with("00:00"))
#aucune info supplémentaire apportée par "+00:00" à la fin de date car toutes
#les lignes finissent par ça, on peut le supprimer
df$date <- substr(df$date,0,nchar(df$date)-6)
df$Date <- as.Date(df$date)
df$Heure <- substr(df$date, 11, 19)


#QUESTIONNAIRE 1
###Durée médiane alerte 
######Création d'un id par alerte
df_clean <- df %>%
  filter(airport_alert_id!="") %>%
  mutate(is_last_lightning_cloud_ground = case_when(
    is_last_lightning_cloud_ground %in% c("True") ~ TRUE,
    is_last_lightning_cloud_ground %in% c("False") ~ FALSE,
      TRUE ~ FALSE
    )
  ) %>%
  arrange(airport, date) %>%
  group_by(airport) %>%
  mutate(
    id_eclair = cumsum(lag(is_last_lightning_cloud_ground, default = FALSE)) + 1
  )

#####Calcul durée de chaque alerte
duree_alerte <- df_clean %>%
  group_by(airport, id_eclair) %>%
  summarise(
    premier_eclair = min(date, na.rm = TRUE),
    dernier_eclair = max(date, na.rm = TRUE),
    nb_eclairs = n(),
    duree_min = if_else(
      nb_eclairs == 1,
      30,
      as.numeric(difftime(dernier_eclair, premier_eclair, units = "mins")) + 30
    ),
    .groups = "drop"
  )

#####Calcul durée médiane
median(duree_alerte$duree_min)

#####Nombre d'alertes orageuses
n_distinct(df_clean$id_eclair)

