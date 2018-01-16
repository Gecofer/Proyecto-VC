# Proyecto final VC

### Autores
- **Braulio Valdivielso Martínez y Gema Correa Fernández**

### Cómo ejecutar

1. Descárguese el fichero de imágenes [aqui](https://consigna.ugr.es/f/HlNB26JIEETwj9HL/images.zip).
2. Créese una carpeta `images` en el mismo directorio que están el archivo README y src y descomprimanse las imagenes dentro.
3. Navegar con la terminal al directorio `src` y ejecutar `python main.py`

Se mostrarán ejemplos de imágenes a las que se les ha hecho proyecciones
cilíndricas, esféricas, se mostrará un mosaico hecho con la proyección
esférica y el algoritmo de Burt-Adelson (además de algunas imágenes
intermedias utilizadas en la memoria para mostrar el funcionamiento
del algoritmo) y finalmente se realizará un mosaico de la Alhambra
en el que se percibirá la diferencia entre usar Burt-Adelson y no usarlo.

### Ficheros:
- util.py: utilidades para visualización de imágenes
- warps.py: contiene la implementación de la proyección esférica y cilíndrica
- burt_adelson.py: contiene la implementación el algoritmo de Burt-Adelson
- mosaic.py: implementación del algoritmo de panorámicas con Burt-Adelson adaptado
- tests.py: algunas pruebas de visualizaciones que hemos desarrollado




