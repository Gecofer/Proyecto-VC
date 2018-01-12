# Proyecto final VC

**IMPORTANTE LEER:**

1. Esta modificado el archivo *burt_adelson.py*, en él hago uso de la función
   para la creación de la Pirámide Laplaciana de https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html,
   y no uso la que creaste. En la funcion *burt_adelson*, la imagen no la normalizo,
   si la normalizas la imagen sale mal. Además, leo las imágenes de la siguiente manera:

   imagen = cv2.imread("../images/imagen.jpg", 1)
   imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

   Conforme aumentemos los niveles en las pirámides, mejor imagen obtenemos, es decir,
   más difuminada y mejor combinación de mezcla. Lo que te envié por Telegram.

   Todo lo que he modificado está en el el archivo *burt_adelson.py*, no he cambiando el *tests.py*

   ** Deberemos comprobar que no tenemos el mismo documento que
   https://github.com/yrevar/semi_automated_cinemagraph/blob/main/blending_utils.py
   por si dice que nos hemos copiado**

2. En el archivo *mosaic.py*, debemos crearnos la máscara ¿pero cómo? Porque con la línea actual
   lo que obtenemos es el liezo negro, y una imagen blanca y eso no nos interesa:

   mask = (canvas > 0).astype(np.uint8) * 255

   Sin embargo, deberíamos usar los pesos, algo así https://docs.opencv.org/3.0.0/d0/d86/tutorial_py_image_arithmetics.html:

   mask = cv2.addWeighted(canvas,0.7,tmp_canvas,0.3,0)

   (Me estoy refiriendo en el mosaico cuando pegamos las imágenes por la izquierda) Entonces no sé
   lo que hacer ya, porque luego el algoritmo de Burt and Adelson, falla, da un error de OpenCV.

   https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/

   http://vision.cse.psu.edu/courses/CompPhoto/pyramidblending.pdf

   http://www.morethantechnical.com/2017/09/29/laplacian-pyramid-with-masks-in-opencv-python/

   A ver si para usar burt adelson, debemos eliminar los bordes negros ¿?. He estado mirado el proyecto
   ya hecho en internet, y lo que hacen es cuando crean el mosaico para dos imágenes, antes de devolver
   el mosaico hecho hacen:

   Mat mask = Mat::zeros(expanded_im1.rows, expanded_im1.cols, CV_32F);

   for (int r = 0; r < mask.rows; r++)
		for (int c = 0; c < mask.cols; c++)
			if (expanded_im1_gray.at<float>(r,c) != 0.0)
				mask.at<float>(r,c) = 1.0;

   Mat mosaic = BurtAdelson(expanded_im1, expanded_im2, mask);

   Todo lo que he modificado está en el el archivo *mosaic.py*, no he tocado el *tests.py*
   Además uso las imágenes de la Alhambra, para ver hacer el mosaico, ya que pienso que los colores
   y la iluminación es más diferente que la de Guernica.

3. He añadido otra imagen para generar el mosaico *flores.jpg*, donde creo que hay buenos cambios de colores.






