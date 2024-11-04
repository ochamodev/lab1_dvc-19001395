# Configuración de DVC S3

Utilice S3 para la configuración de mi proyecto, junto con el dataset de la demo de DVC
para la implementación de este laboratorio.

Deberan crear un bucket de S3 para poder usarlo.

Estos son los comandos a usar:
`dvc remote add -d myremote s3://<bucket>/<key>`

Configurar el access key

`dvc remote modify --local myremote access_key_id 'mysecret'`
`dvc remote modify --local myremote secret_access_key 'mysecret'`


Dentro de `params.yaml` encontraran los parametros que se usaron.
Dentro de dvc.yaml todas las stages definidas.

Ejecutar `dvc repro` Para poder visualizar la ejecución de todo el pipeline :).


# Nota: Se intentara adjuntar el resultado de cada ejecución 