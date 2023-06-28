# LM-data-gen
Upload the LMsyn.py to your colab runtime.
Import the model as "From LMsyn import LMmodel"
Set the batch size, epochs and pass the dataframe for training 
"
model = LMmodel( batch_size=8, epochs=1)
model.fit(x_train)
"
To generate sample data, use "model.sample()" function. There are 3 arguments for this function,
n_samples: number of rows to be generated,
cond_col: name column tobe conditioned
cond: conditions,
if the column has continous values pass as list of all possible values for example "cond_col='ATC' , cond=[1.6,13.5,5.2]"
if the column is discrete pass a dictionary in format of
"{class1: weight1,class2: weight2, class3: weight3 }" and so on.
"class" is the discrete value  and "weight" is used to weigh the possibility for each class.
An example of condition "cond_col='Gender', cond={'male': 2,'female':1}"
Please note weight of class should always be integer value

I have attached a colab notebook as example
