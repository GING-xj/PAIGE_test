//
// Created by xjxj on 19-1-22.
//
// Train RF model with all *.csv files under a specific folder

#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <iostream>
#include <string>
#include <vector>

#include "PAIGE_Loss.h"
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace shark;
using namespace boost::filesystem;


int main(int argc,char ** argv) {

    if(argc!=3)
    {
        cout<<"Usage:"<<endl;
        cout<<"~/TrainModel /dir/to/first/training/dataset model_name"<<endl
            <<"eg. ~/TrainModel /home/Data/  random_forest"<<endl;
        return EXIT_FAILURE;
    }

    string dir_to_csv_files=argv[1];
    string model_name=argv[2];
    model_name=model_name+".model";
    vector<string> csv_files_vec;

    path csv_files_path(dir_to_csv_files);
    if(!exists(csv_files_path))
    {
        return EXIT_FAILURE;
    }

    directory_iterator end_iter;

    for(directory_iterator iter(csv_files_path);iter!=end_iter;++iter)
    {
        if(is_regular(iter->status()))
        {
            csv_files_vec.push_back(iter->path().string());
        }
    }

    ClassificationDataset data;


    for(const auto str:csv_files_vec)
    {
        path csv_file_path(str);
        string extension_name=csv_file_path.extension().string();

        if(!(extension_name==".csv"))
            continue;

        ClassificationDataset temp_data;
        importCSV(temp_data,str.c_str(),FIRST_COLUMN);
        data.append(temp_data);
    }

    cout << "Training set - number of data points: " << data.numberOfElements()
         << " number of classes: " << numberOfClasses(data)
         << " input dimension: " << inputDimension(data) << endl;

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


