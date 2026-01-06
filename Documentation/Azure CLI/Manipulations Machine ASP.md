# Guide sur manipulation de Azure CLI sur notre instance Azure Web App (ASP)

# Créer une machine ASP (App Service Plan) via Azure CLI

```
# Variables
RG=mon-groupe-de-ressources
LOCATION=westeurope
ASP_NAME=mon-asp
APP_NAME=mon-app

# Connexion et sélection d'abonnement si besoin
az login --use-device-code
# az account set --subscription "<id-ou-nom-abonnement>"

# 1) Groupe de ressources (indispensale pour y affecter votre ASP)
az group create -n $RG -l $LOCATION

# 2) App Service Plan (Linux)
az appservice plan create -n $ASP_NAME -g $RG --sku B1 --is-linux
# Variante Windows : supprimer --is-linux

# 3) Web App associée
az webapp create -g $RG -p $ASP_NAME -n $APP_NAME --runtime "PYTHON:3.10"
```

# Commandes utiles ASP :

## Lister les ASP du groupe : 
```
az appservice plan list -g $RG -o table
```

## Voir le détail d'un ASP : 
```
az appservice plan show -g $RG -n $ASP_NAME
```

## Changer le SKU / la taille : 
```
az appservice plan update -g $RG -n $ASP_NAME --sku P1v2
```

## Supprimer un ASP : 
```
az appservice plan delete -g $RG -n $ASP_NAME
```

## Redémarrer une ASP
```
az webapp restart -g DefaultResourceGroup-CCAN -n $ASP_NAME
```


# Connexion à notre application (Air Paradis) depuis AzureCLI

```
# Commande de connexion à la machine
kamel [ ~ ]$ az webapp ssh -g DefaultResourceGroup-CCAN -n AirParadisAPI

# Vérifier en inspéctant l'emplacement et le contenu de l'ASP
(antenv) root@1b630ce9b28d:/tmp/8de3bc13861aee5# pwd
/tmp/8de3bc13861aee5
(antenv) root@1b630ce9b28d:/tmp/8de3bc13861aee5# ls -a
. .. antenv app requirements.txt
```
