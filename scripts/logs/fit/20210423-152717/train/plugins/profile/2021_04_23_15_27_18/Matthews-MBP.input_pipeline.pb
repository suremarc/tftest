	?L?n?@?L?n?@!?L?n?@	g kF??g kF??!g kF??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?L?n?@? l@????A?????@Y/?H???rEagerKernelExecute 0*	|?G?^n@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????Q???!X?[??>K@)#?#????1&?҉??H@:Preprocessing2U
Iterator::Model::ParallelMapV2b????k??!Ok?ڝ=@)b????k??1Ok?ڝ=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??v1͔?!??l?ĸ @)?/?$??1Ax?@?^@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicePqx?܉?!??I2?@)Pqx?܉?1??I2?@:Preprocessing2F
Iterator::Model ??c??!sʤЌ?@@)?5?e܄?18y???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip4H?Sȕ??!ǚ??9?P@)-C??6z?1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??[?d8n?!?.??K??)??[?d8n?1?.??K??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??k*??!?6p?K@)()? ?\?1???P???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9g kF??Id?S??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? l@????? l@????!? l@????      ??!       "      ??!       *      ??!       2	?????@?????@!?????@:      ??!       B      ??!       J	/?H???/?H???!/?H???R      ??!       Z	/?H???/?H???!/?H???b      ??!       JCPU_ONLYYg kF??b qd?S??X@