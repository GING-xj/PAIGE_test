//
// Created by xjxj on 19-1-17.
//
// Train a random forest classifier with single csv file

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
    if(argc!=3)
    {
        cout<<"Usage:"<<endl;
        cout<<"~/TrainModel /path/to/training/dataset model_name"<<endl
            <<"eg. ~/TrainModel /home/Data/data.csv svm"<<endl;
        return EXIT_FAILURE;
    }

    string training_data_path;
    string model_name;

    training_data_path=argv[1];
    model_name=argv[2];
    model_name=model_name+".model";

    ClassificationDataset data;
    importCSV(data,training_data_path.c_str(),FIRST_COLUMN);

    cout << "Training set - number of data points: " << data.numberOfElements()
         << " number of classes: " << numberOfClasses(data)
         << " input dimension: " << inputDimension(data) << endl;

    RFTrainer trainer;
    RFClassifier model;

    trainer.setNTrees(50);
    trainer.setNodeSize(3);

    trainer.train(model, data);

    cout<<"Saving model to disk"<<endl;

    ofstream ofs(model_name.c_str());
    boost::archive::text_oarchive oa(ofs);
    model.write(oa);
    ofs.close();

    cout<<"Done"<<endl;

    return EXIT_SUCCESS;
}

