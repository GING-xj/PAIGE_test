//
// Created by xjxj on 19-1-17.
//
// Train RF model from two csv file


#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <iostream>
#include <string>

#include "PAIGE_Loss.h"
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace shark;


int main(int argc,char ** argv) {

    if(argc!=4)
    {
        cout<<"Usage:"<<endl;
        cout<<"~/TrainModel /path/to/first/training/dataset /path/to/second/training/dataset model_name"<<endl
            <<"eg. ~/TrainModel /home/Data/data1.csv /home/Data/data2.csv random_forest"<<endl;
        return EXIT_FAILURE;
    }

    string training_data_path_1;
    string training_data_path_2;
    string model_name;

    training_data_path_1=argv[1];
    training_data_path_2=argv[2];
    model_name=argv[3];
    model_name=model_name+".model";

    ClassificationDataset data;
    importCSV(data,training_data_path_1.c_str(),FIRST_COLUMN);


    ClassificationDataset another_data ;
    importCSV(another_data,training_data_path_2.c_str(),FIRST_COLUMN);

    cout << "Training set - number of data points: " << data.numberOfElements()
         << " number of classes: " << numberOfClasses(data)
         << " input dimension: " << inputDimension(data) << endl;

    cout << "Second Training set - number of data points: " << another_data.numberOfElements()
         << " number of classes: " << numberOfClasses(another_data)
         << " input dimension: " << inputDimension(another_data) << endl;

    //Generate a random forest
    //###begin<train>

    data.append(another_data);


    RFTrainer trainer;
    RFClassifier model;

    trainer.setNTrees(50);
    trainer.setNodeSize(3);

    trainer.train(model, data);

    unsigned int cnt{0};
    Data<unsigned int > labels=data.labels();
    unsigned int number_of_0{0},number_of_1{0};
    double ratio;
    for(size_t i=0;i<labels.numberOfElements();++i)
    {
        ++cnt;
        if(labels.element(i)==0)
        {
            ++number_of_0;
        }
        else if(labels.element(i)==1)
        {
            ++number_of_1;
        }
        else
        {
            cout<<"Error!"<<endl;
        }
    }

    ratio=(double)number_of_0/(double)number_of_1;
    cout<<"Number of 0:"<<number_of_0<<endl
        <<"Number of 1:"<<number_of_1<<endl
        <<"Ratio:"<<ratio<<endl
        <<"Total data points:"<<cnt<<endl;

    ofstream ofs(model_name.c_str());
    boost::archive::text_oarchive oa(ofs);
    model.write(oa);
    ofs.close();


    return EXIT_SUCCESS;
}


