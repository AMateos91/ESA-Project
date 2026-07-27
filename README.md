# Fire Detection AI - MODIS + AEMET

Sistema de detección de incendios forestales mediante imágenes satelitales MODIS, datos meteorológicos AEMET y Deep Learning.

El objetivo del proyecto es desarrollar un modelo capaz de generar mapas de probabilidad de incendio combinando información espectral satelital y variables meteorológicas.

---

## Arquitectura

NASA Earthdata
      |
      |
  Productos MODIS
      |
      |----------------------|
      |                      |
   MOD09GA              MOD14A1
 Reflectancia          Fire Mask
      |                      |
      |----------------------|
             |
             |
       Preprocesamiento
             |
             |
        Datos MODIS
             |
             |
        AEMET API
             |
             |
    Fusión espacial de datos
             |
             |
          U-Net
             |
             |
   Mapa de probabilidad
      de incendios 
