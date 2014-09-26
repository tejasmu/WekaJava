package com.company;

import java.io.*;
import java.util.*;

import com.mongodb.*;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.instance.Randomize;

public class Main{

    public static void main(String args[]) throws Exception
    {

        try{

            //ArffFileStatus();

            String inArff = "F:\\Mani\\Experiments\\RawArff\\TejasTry1.arff";
            //String outputpath = "F:\\Mani\\Experiments\\SmoteArff\\mutualuniremoved_usermodel1.arff";
            String outSmote = "F:\\Mani\\Experiments\\SmoteArff\\TejasSmoteTry1.arff";
            
            //String logisticregressionmodel = "F:\\Mani\\Experiments\\RegressionModels\\logistic_mfmodel1.model");
            String LRModel = "F:\\Mani\\Experiments\\RegressionModels\\TejasLogisticTry1_1.model";
        
            double percentage = 15300.00;
            
            //GetTrainStatus(outSmote);

            //StoreLogisticModel(outSmote,LRModel);

            //String path = "/home/mvijaya2/Experiments/Models/RawArff/usermodel_1.arff";
            //GetTrainStatus(path);

            //StoreLogisticModel();

            //CreateTrainingDataset();
            
            //ApplyFilter(inArff,outSmote,percentage);
            
            StoreLogisticModel(outSmote,LRModel);
            
            //TrainingFeatureExperiments();

            ClassifytestData();

            PrintModel("F:\\Mani\\Experiments\\RegressionModels\\TejasLogisticTry1_1"
            		+ ".model");

            //StoreModelsforDataset();

            //PrintAllModels();

            //TestDataModified();

            //ClassifyNoMutual();


        }
        catch (Exception ex)
        {
            System.out.print(ex);
        }
        System.out.println("Done");
    }

    public static void ArffFileStatus()
    {
        String tweetidstatic = "";
        try
        {
                List<String> usernamelist = Arrays.asList("cnwatanabe","pintendo64","smolix","syv_k","thebraco","toybabychildren","xmaz83","MUceny","LTock","LifeWithLaughs","SMGebru","drbrady12","thisparticular");           
                for(String username : usernamelist) {
                int totalpositivecounter = 0;
                int totalnegativecounter = 0;

                Mongo mongoClient = new Mongo("localhost", 27017);
                DB querydb = mongoClient.getDB("userdatasetQueries");
                String queryCollecName = username + "_query";
                DBCollection coll = querydb.getCollection(queryCollecName);
                DBCursor cursor = coll.find();
                int counter = 1;
                try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter( "F:\\Mani\\Experiments\\FileStats.txt", true))))
                {
                    out.println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=" + username);
                    while (cursor.hasNext())
                    {
                        DBObject doc = cursor.next();
                        BasicDBList basicDBList = (BasicDBList) ((BasicDBObject) doc).get(username);
                        for (Object bquery : basicDBList)
                        {
                            int querypositivecounter = 0;
                            int querynegativecounter = 0;
                            boolean positiveflag = true;
                            boolean negativeflag = true;
                            String tweetid = ((BasicDBObject) bquery).get("id").toString();
                            tweetidstatic = tweetid;
                            String path = "F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\ResultArff" + username + "\\" + tweetid + ".arff";
                            Instances data = new Instances(new BufferedReader(new FileReader(path)));
                            for (int i = 0; i < data.numInstances(); i++)
                            {
                                Instance isnt = data.instance(i);
//                                if (isnt.value(9) == 1.0)
//                                {
//                                    querypositivecounter++;
//                                }
//                                else if (isnt.value(9) == 0.0)
//                                {
//                                    querynegativecounter++;
//                                }
                                if (isnt.value(9) == 1.0 && positiveflag)
                                {
                                    totalpositivecounter++;
                                    positiveflag = false;
                                }
                                else if (isnt.value(9) == 0.0 && negativeflag)
                                {
                                    totalnegativecounter++;
                                    negativeflag = false;
                                }
                            }
                            //out.println("Query Info: " + tweetid + "  +ve " + querypositivecounter+ " -ve " + querynegativecounter );
                        }
                    }
                    out.println("UserStats: " + username + "  +ve " + totalpositivecounter + " -ve " + totalnegativecounter);
                    out.println();
                }

                System.out.println("Done for: " + username );
            }
        }
        catch (Exception ex)
        {
            System.out.println(tweetidstatic);
            ex.printStackTrace();
        }
    }

    public static void GetTrainStatus(String path)
    {
        try
        {
            //String path = "/home/mvijaya2/ThesisCodeBase/TrainingDataset/6user-50Queries/smote_fiftytrain4.arff";
            int positivecounter = 0;
            int negativecounter = 0;
            Instances data = new Instances(new BufferedReader(new FileReader(path)));
            for (int i = 0; i < data.numInstances(); i++)
            {
                Instance isnt = data.instance(i);
                if (isnt.value(9) == 1.0)
                {
                    positivecounter++;
                }
                else if (isnt.value(9) == 0.0)
                {
                    negativecounter++;
                }
            }
            System.out.println("Positive Samples:" + positivecounter);
            System.out.println("Negative Samples:" + negativecounter);
            System.out.println("All Samples:" + data.numInstances());
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

    }

    public static void TrainingFeatureExperiments()
    {
        try
        {
            String arfffilepath = "F:\\Mani\\Experiments\\SmoteArff\\mutualuniremoved_usermodel1.arff";
            String sourcearfffilepath = "F:\\Mani\\Experiments\\SmoteArff\\smote_usermodel1.arff";
            
            //String arfffilepath = "F:\\Mani\\Experiments\\SmoteArff\\TheFunnyKent.arff";
            //String sourcearfffilepath = "F:\\Mani\\Experiments\\SmoteArff\\TheFunnyKent.arff";
            try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(arfffilepath, true))))
            {
                Instances data = new Instances(new BufferedReader(new FileReader(sourcearfffilepath)));
                for (int i = 0; i < data.numInstances(); i++)
                {
                    Instance isnt = data.instance(i);
                    //out.println(isnt.value(5) + "," + (int) isnt.value(9));
                    out.println(isnt.value(0) + "," + isnt.value(1) + "," + isnt.value(2) + "," + isnt.value(3) + "," + isnt.value(4) + ","  + isnt.value(7) + "," +  (int) isnt.value(9));
                }
            }
            System.out.println("Training Features Experiments complete");
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }

    }

    public static void TestDataModified()
    {
        try
        {
            List<String> usernamelist = Arrays.asList("CAPYGAMES","CSLewisDaily","GulfstreamPark","Pointshelf","RadegundBeers","MichaelSkolnik","bhorowitz","craigdadams","dannyhengel","dhh","SheThrivesNet","drockney","entylawyer","haran","dennydermanto","marc_lepage","mfsprouse","miltonbrewery","theprinceoliver","timnelms","netbull11","thatkeith","mylongtham","theTimoyer","Daniel_Rubino","Fatma962ibrahim","PulledGoalie","jonnykanyon","totalbetchmove","NitaTyndall","EssenPaige","MELISSAMARIE","Schwabe16","sara_evelyn36","ThatsFahney","prasannalaldas","pixelmushr00m","MissaAlvarado","antonylittle","JimmyFinn3","LemonsAubree","eddie_reedy","randy_moss_TV","nhmaaske","noahtheasian","Jeffrey_Heim");

            for(String username : usernamelist) {

                Mongo mongoClient = new Mongo("localhost", 27017);
                DB querydb = mongoClient.getDB("userdatasetQueries");
                String queryCollecName = username + "_query";
                DBCollection coll = querydb.getCollection(queryCollecName);
                DBCursor cursor = coll.find();

                    while (cursor.hasNext())
                    {
                        DBObject doc = cursor.next();
                        BasicDBList basicDBList = (BasicDBList) ((BasicDBObject) doc).get(username);
                        for (Object bquery : basicDBList)
                        {
                            String tweetid = ((BasicDBObject) bquery).get("id").toString();
                            String path = "F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\ResultArff\\" + username + "\\" + tweetid + ".arff";
                            String arfffilepath = "F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\ResultArff\\" + username + "\\" + tweetid + "_nomf.arff";
                            Instances data = new Instances(new BufferedReader(new FileReader(path)));
                            try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(arfffilepath, true))))
                            {
                                out.println("@relation ctxbot");
                                out.println("");
                                out.println("@attribute mutualfriendrank numeric");
                                out.println("@attribute classy {0,1}");
                                out.println("");
                                out.println("");
                                out.println("@data");

                                for (int i = 0; i < data.numInstances(); i++) {
                                    Instance isnt = data.instance(i);
                                    out.println(isnt.value(5) + "," + (int) isnt.value(9));
                                    //out.println(isnt.toString());
                                }
                            }
                        }
                    }

                System.out.println("Done for: " + username );
            }

        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }

    }

    public static void CreateTrainingDataset()
    {
        try
        {
            List<String> usernamelist = Arrays.asList("AlgebraFact","Anistuffs","AshleyUSA","ProfNoodlearms","cnwatanabe","pintendo64","smolix","syv_k","thebraco","toybabychildren","xmaz83","MUceny","LTock","LifeWithLaughs","SMGebru","drbrady12","thisparticular");
        	//List<String> usernamelist = Arrays.asList("AlgebraFact","Anistuffs","AshleyUSA","ProfNoodlearms","cnwatanabe");
        	//List<String> usernamelist = Arrays.asList("TheFunnyKent");
        	//String arfffilepath = "F:\\Mani\\Experiments\\RawArff\\usermodel_1.arff";
        	
        	String arfffilepath = "F:\\Mani\\Experiments\\RawArff\\TejasTry1.arff";

            for(String username : usernamelist) {

                Mongo mongoClient = new Mongo("localhost", 27017);
                DB querydb = mongoClient.getDB("userdatasetQueries");
                String queryCollecName = username + "_query";
                DBCollection coll = querydb.getCollection(queryCollecName);
                DBCursor cursor = coll.find();
                int counter = 1;
                try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(arfffilepath, true))))
                {
                    while (cursor.hasNext())
                    {
                        DBObject doc = cursor.next();
                        BasicDBList basicDBList = (BasicDBList) ((BasicDBObject) doc).get(username);
                        for (Object bquery : basicDBList)
                        {
                            if (counter > 20){
                                break;
                            }
                            String tweetid = ((BasicDBObject) bquery).get("id").toString();
                            String path = "F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\ResultArff\\" + username + "\\" + tweetid + ".arff";
                            Instances data = new Instances(new BufferedReader(new FileReader(path)));
                            for (int i = 0; i < data.numInstances(); i++)
                            {
                                Instance isnt = data.instance(i);
                                out.println(isnt.toString());
                            }
                            counter++;
                        }
                    }
                }
                System.out.println("Done for: " + username );
            }
            System.out.println("Created TejasTry1.arff");

        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }

    }

    public static void ClassifytestData(){
        try
        {
            //List<String> usernamelist = Arrays.asList("netbull11","thatkeith","mylongtham","theTimoyer","Daniel_Rubino","Fatma962ibrahim","PulledGoalie","jonnykanyon","totalbetchmove","NitaTyndall","EssenPaige","MELISSAMARIE","Schwabe16","sara_evelyn36","ThatsFahney","prasannalaldas","pixelmushr00m","MissaAlvarado","antonylittle","JimmyFinn3","LemonsAubree","eddie_reedy","randy_moss_TV","nhmaaske","noahtheasian","Jeffrey_Heim");
            //List<String> usernamelist = Arrays.asList("MichaelSkolnik","bhorowitz","craigdadams","dannyhengel","dhh","SheThrivesNet","drockney","entylawyer","haran","dennydermanto","marc_lepage","mfsprouse","miltonbrewery","theprinceoliver","timnelms");
            List<String> usernamelist = Arrays.asList("CAPYGAMES","CSLewisDaily","GulfstreamPark","Pointshelf","RadegundBeers");
            for(String username : usernamelist)
            {

                try {
                    int totalpositivecounter = 0;
                    Mongo mongoClient = new Mongo("localhost", 27017);
                    DB querydb = mongoClient.getDB("userdatasetQueries");
                    //DB indexresultdb = mongoClient.getDB("userdatasetJuneWekaIndex");
                    //DB indexresultdb = mongoClient.getDB("userdatasetNewWekaIndex");
                    DB indexresultdb = mongoClient.getDB("userdatasetWekaIndex");
                    DB finalresultdb = mongoClient.getDB("userModel1_MySystem");

                    String queryCollecName = username + "_query";
                    DBCollection coll = querydb.getCollection(queryCollecName);
                    DBCursor cursor = coll.find();

                    //ObjectInputStream ois = new ObjectInputStream(new FileInputStream("F:\\Mani\\Experiments\\RegressionModels\\logistic_usermodel_1.model"));
                    ObjectInputStream ois = new ObjectInputStream(new FileInputStream("F:\\Mani\\Experiments\\RegressionModels\\TejasLogisticTry1.model"));
                    Logistic cls = (Logistic) ois.readObject();
                    ois.close();
                    int classcounter = 0;

                    while (cursor.hasNext())
                    {
                        DBObject doc = cursor.next();
                        BasicDBList basicDBList = (BasicDBList) ((BasicDBObject) doc).get(username);
                        for (Object bquery : basicDBList) {
                            boolean positiveflag = true;
                            String tweetid = ((BasicDBObject) bquery).get("id").toString();
                            String tweettext = ((BasicDBObject) bquery).get("text").toString();
                            Object entitylist = ((BasicDBObject) bquery).get("entities");
                            Object hashtaglist = ((BasicDBObject) entitylist).get("hashtags");
                            Object hashtagindex = ((BasicDBList) hashtaglist).get(0);
                            String qhahstag = ((BasicDBObject) hashtagindex).get("text").toString();

                            DBCollection resultcoll = finalresultdb.getCollection(tweetid + "_classified");
                            DBCollection collindex = indexresultdb.getCollection(tweetid + "_wekaindex");

                            HashMap<Integer, String> tagindexhashmap = new HashMap<Integer, String>();
                            HashMap<Integer, String> tweetidindexhashmap = new HashMap<Integer, String>();
                            DBCursor cursorindex = collindex.find();

                            while (cursorindex.hasNext()) {
                                DBObject docobj = cursorindex.next();
                                Integer index = (Integer) ((BasicDBObject) docobj).get("index");
                                String hashtag = (String) ((BasicDBObject) docobj).get("tag");
                                String tid = ((BasicDBObject) docobj).get("tweetid").toString();
                                tagindexhashmap.put(index, hashtag);
                                tweetidindexhashmap.put(index, tid);
                            }

                            String path = "F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\ResultArff\\" + username + "\\" + tweetid + ".arff";

                            Instances data = new Instances(new BufferedReader(new FileReader(path)));
                            int localindex = 1;
                            boolean flag = true;

                            for (int i = 0; i < data.numInstances(); i++)
                            {
                                Instance isnt = data.instance(i);
                                if (isnt.value(9) == 1.0 && positiveflag)
                                {
                                    totalpositivecounter++;
                                    positiveflag = false;
                                }
                                data.setClassIndex(isnt.numAttributes() - 1);
                                double[] values = cls.distributionForInstance(isnt);

                                double classy = cls.classifyInstance(isnt);
                                if (flag == true) {
                                    if (classy == 1.0) {

                                        if (classy == isnt.value(9)) {
                                            //System.out.println("Class 1");
                                            classcounter++;
                                            flag = false;
                                        }
                                    }
                                }

                                BasicDBObject insertdoc = new BasicDBObject("index", localindex).
                                        append("hashtag", tagindexhashmap.get(localindex)).
                                        append("classpredicted", classy).
                                        append("tweetid", tweetidindexhashmap.get(localindex)).
                                        append("orginalclass", isnt.value(9)).
                                        append("onepred", values[1]).
                                        append("zeropred", values[0]);
                                //                            append("simrank", isnt.value(0)).
                                //                            append("timerank", isnt.value(1)).
                                //                            append("trendrank", isnt.value(2)).
                                //                            append("atmentionrank", isnt.value(3)).
                                //                            append("favscorerank", isnt.value(4)).
                                //                            append("mutualfriendrank", isnt.value(5)).
                                //                            append("mutualfollowerrank", isnt.value(6)).
                                //                            append("commonhahashtagrank", isnt.value(7)).
                                //                            append("unibirank", isnt.value(8));
                                resultcoll.insert(insertdoc);
                                localindex++;
                            }
                            //System.out.println("Done" + localindex + "/" + tweetid + "/" + classcounter);
                        }
                        System.out.println("Username " + username + " Correctly: " + classcounter + " MaxPossible " + totalpositivecounter);
                    }
                }
                catch (Exception ex) {
                    System.out.println(ex);
                    ex.printStackTrace();
                }
            }
            System.out.println("Done");
        }
        catch (Exception ex) {
            System.out.println(ex);
            ex.printStackTrace();
        }
    }

    public static void ClassifyNoMutual(){
        String tweetidinfo = "";
        try
        {
            //List<String> usernamelist = Arrays.asList("thatkeith","mylongtham","theTimoyer","Daniel_Rubino","Fatma962ibrahim","PulledGoalie","jonnykanyon","totalbetchmove","NitaTyndall","EssenPaige","MELISSAMARIE","Schwabe16","sara_evelyn36","ThatsFahney","prasannalaldas","pixelmushr00m","MissaAlvarado","antonylittle","JimmyFinn3","LemonsAubree","eddie_reedy","randy_moss_TV","nhmaaske","noahtheasian","Jeffrey_Heim");
            //List<String> usernamelist = Arrays.asList("netbull11","thatkeith","mylongtham","theTimoyer","Daniel_Rubino","Fatma962ibrahim","PulledGoalie","jonnykanyon","totalbetchmove","NitaTyndall","EssenPaige","MELISSAMARIE","Schwabe16","sara_evelyn36","ThatsFahney","prasannalaldas","pixelmushr00m","MissaAlvarado","antonylittle","JimmyFinn3","LemonsAubree","eddie_reedy","randy_moss_TV","nhmaaske","noahtheasian","Jeffrey_Heim");
            //List<String> usernamelist = Arrays.asList("MichaelSkolnik","bhorowitz","craigdadams","dannyhengel","dhh","SheThrivesNet","drockney","entylawyer","haran","dennydermanto","marc_lepage","mfsprouse","miltonbrewery","theprinceoliver","timnelms");
            //List<String> usernamelist = Arrays.asList("CAPYGAMES","CSLewisDaily","GulfstreamPark","Pointshelf","RadegundBeers");
            
        	List<String> usernamelist = Arrays.asList("TheFunnyKent");
        	
        	for(String username : usernamelist)
            {

                try {
                    int totalpositivecounter = 0;
                    Mongo mongoClient = new Mongo("localhost", 27017);
                    DB querydb = mongoClient.getDB("userdatasetQueries");
                    DB indexresultdb = mongoClient.getDB("userdatasetJuneWekaIndex");
                    //DB indexresultdb = mongoClient.getDB("userdatasetNewWekaIndex");
                    //DB indexresultdb = mongoClient.getDB("userdatasetWekaIndex");
                    DB finalresultdb = mongoClient.getDB("onlymf_MySystem");

                    String queryCollecName = username + "_query";
                    DBCollection coll = querydb.getCollection(queryCollecName);
                    DBCursor cursor = coll.find();

                    //ObjectInputStream ois = new ObjectInputStream(new FileInputStream("F:\\Mani\\Experiments\\RegressionModels\\logistic_mfmodel1.model"));
                    
                    ObjectInputStream ois = new ObjectInputStream(new FileInputStream("F:\\Mani\\Experiments\\RegressionModels\\TheFunnyKent.model"));
                    Logistic cls = (Logistic) ois.readObject();
                    ois.close();
                    int classcounter = 0;

                    while (cursor.hasNext())
                    {
                        DBObject doc = cursor.next();
                        BasicDBList basicDBList = (BasicDBList) ((BasicDBObject) doc).get(username);
                        for (Object bquery : basicDBList) {
                            boolean positiveflag = true;
                            String tweetid = ((BasicDBObject) bquery).get("id").toString();
                            String tweettext = ((BasicDBObject) bquery).get("text").toString();
                            Object entitylist = ((BasicDBObject) bquery).get("entities");
                            Object hashtaglist = ((BasicDBObject) entitylist).get("hashtags");
                            Object hashtagindex = ((BasicDBList) hashtaglist).get(0);
                            String qhahstag = ((BasicDBObject) hashtagindex).get("text").toString();

                            DBCollection resultcoll = finalresultdb.getCollection(tweetid + "_classified");
                            DBCollection collindex = indexresultdb.getCollection(tweetid + "_wekaindex");

                            HashMap<Integer, String> tagindexhashmap = new HashMap<Integer, String>();
                            HashMap<Integer, String> tweetidindexhashmap = new HashMap<Integer, String>();
                            DBCursor cursorindex = collindex.find();

                            while (cursorindex.hasNext()) {
                                DBObject docobj = cursorindex.next();
                                Integer index = (Integer) ((BasicDBObject) docobj).get("index");
                                String hashtag = (String) ((BasicDBObject) docobj).get("tag");
                                String tid = ((BasicDBObject) docobj).get("tweetid").toString();
                                tagindexhashmap.put(index, hashtag);
                                tweetidindexhashmap.put(index, tid);
                            }
                            tweetidinfo = tweetid.toString();
                            String path = "F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\ResultArff\\" + username + "\\" + tweetid + "_nomf.arff";

                            Instances data = new Instances(new BufferedReader(new FileReader(path)));
                            int localindex = 1;
                            boolean flag = true;

                            for (int i = 0; i < data.numInstances(); i++)
                            {
                                Instance isnt = data.instance(i);
                                if (isnt.value(1) == 1.0 && positiveflag)
                                {
                                    totalpositivecounter++;
                                    positiveflag = false;
                                }
                                data.setClassIndex(isnt.numAttributes() - 1);
                                double[] values = cls.distributionForInstance(isnt);

                                double classy = cls.classifyInstance(isnt);
                                if (flag == true) {
                                    if (classy == 1.0) {

                                        if (classy == isnt.value(1)) {
                                            //System.out.println("Class 1");
                                            classcounter++;
                                            flag = false;
                                        }
                                    }
                                }

                                BasicDBObject insertdoc = new BasicDBObject("index", localindex).
                                        append("hashtag", tagindexhashmap.get(localindex)).
                                        append("classpredicted", classy).
                                        append("tweetid", tweetidindexhashmap.get(localindex)).
                                        append("orginalclass", isnt.value(1)).
                                        append("onepred", values[1]).
                                        append("zeropred", values[0]);
                                //                            append("simrank", isnt.value(0)).
                                //                            append("timerank", isnt.value(1)).
                                //                            append("trendrank", isnt.value(2)).
                                //                            append("atmentionrank", isnt.value(3)).
                                //                            append("favscorerank", isnt.value(4)).
                                //                            append("mutualfriendrank", isnt.value(5)).
                                //                            append("mutualfollowerrank", isnt.value(6)).
                                //                            append("commonhahashtagrank", isnt.value(7)).
                                //                            append("unibirank", isnt.value(8));
                                resultcoll.insert(insertdoc);
                                localindex++;
                            }
                            //System.out.println("Done" + localindex + "/" + tweetid + "/" + classcounter);
                        }
                        System.out.println("Username " + username + " Correctly: " + classcounter + " MaxPossible " + totalpositivecounter);
                    }
                }
                catch (Exception ex) {
                    System.out.println(tweetidinfo);
                    System.out.println(ex);
                    ex.printStackTrace();
                }
            }
            System.out.println("Done");
        }
        catch (Exception ex) {
            System.out.println(tweetidinfo);
            System.out.println(ex);
            ex.printStackTrace();
        }
    }

    public static void PrintAllModels()
    {
        //List<Double> percentages = Arrays.asList(100.0,200.0,300.0,400.0,500.0,600.0);
        List<Double> percentages = Arrays.asList(500.0);
        for(Double percent : percentages) {
            String filepath =  "F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\Weka\\Wekanewmodel\\model_" + percent.toString() + "data.model";
            PrintModel(filepath);
        }
    }

    public static void PrintModel(String filepath)
    {   try
        {
            ObjectInputStream ois = new ObjectInputStream(
                    new FileInputStream(filepath));
            Logistic cls = (Logistic) ois.readObject();
            ois.close();
            System.out.println(cls);
        }
        catch (Exception ex)
        {
            System.out.println(ex);
        }

    }

    public static void ApplyFilter(String Path,String outputpath,double percentage) {

        try{
            Instances data = new Instances(new BufferedReader(new FileReader(Path)));
            System.out.println("Started Reading Data");
            data.setClassIndex(data.numAttributes() - 1);
            List<Double> percentages = Arrays.asList(percentage);
            for(Double percent : percentages)
            {
                SMOTE smotefilter = new SMOTE();
                smotefilter.setInputFormat(data);
                smotefilter.setPercentage(percent);
                smotefilter.setClassValue("0");
                System.out.println("Started Filter");
                Instances newData = Filter.useFilter(data,smotefilter);
                System.out.println("Applied Filter");
                Randomize randomizefilter = new Randomize();
                randomizefilter.setInputFormat(newData);

                Instances newRandomizedData = Filter.useFilter(newData,randomizefilter);
                System.out.println("Saved for" + percent.toString());
                ArffSaver saver = new ArffSaver();
                saver.setInstances(newRandomizedData);
                saver.setFile(new File(outputpath));
                saver.writeBatch();
            }
        }
        catch (Exception ex) {
        	ex.printStackTrace();
        }

    }

    public static void ClassifytestDataandPrint(){
        try
        {
            //List<String> usernamelist = Arrays.asList("AlgebraFact", "Anistuffs", "AshleyUSA", "BaoPhotography", "CAPYGAMES", "CSLewisDaily", "GaryContessa", "GulfstreamPark", "LTock", "LifeWithLaughs", "MichaelSkolnik", "Pointshelf", "ProfNoodlearms", "RadegundBeers", "SMGebru");
            //List<String> usernamelist = Arrays.asList("AlgebraFact", "AshleyUSA", "BaoPhotography", "CAPYGAMES", "CSLewisDaily", "GaryContessa", "GulfstreamPark", "LTock", "LifeWithLaughs", "MichaelSkolnik", "Pointshelf", "ProfNoodlearms", "RadegundBeers", "SMGebru");
            //List<String> usernamelist = Arrays.asList("AlgebraFact", "AshleyUSA", "CAPYGAMES", "CSLewisDaily", "GulfstreamPark", "LTock", "LifeWithLaughs", "Pointshelf", "ProfNoodlearms", "RadegundBeers", "SMGebru");
            //List<String> usernamelist = Arrays.asList("AlgebraFact", "Anistuffs", "AshleyUSA", "BaoPhotography", "CAPYGAMES", "CSLewisDaily", "GulfstreamPark", "LTock", "LifeWithLaughs", "MichaelSkolnik", "Pointshelf", "ProfNoodlearms", "RadegundBeers", "SMGebru", "SenatorBoxer", "SheThrivesNet");
            List<String> usernamelist = Arrays.asList("AlgebraFact");
            //Iterate through the username
            for(String username : usernamelist) {

                Mongo mongoClient = new Mongo("localhost", 27017);
                DB querydb = mongoClient.getDB("userdatasetQueries");
                //DB finalresultdb = mongoClient.getDB("userdatasetLogisticResults");
                DB indexresultdb = mongoClient.getDB("userdatasetWekaIndex");

                //for(Double percent : percentages) {

                String queryCollecName = username + "_query";
                DBCollection coll = querydb.getCollection(queryCollecName);
                DBCursor cursor = coll.find();
                HashMap<Integer,Integer> kresultmap = new HashMap<>();
                //ObjectInputStream ois = new ObjectInputStream(new FileInputStream( "/Users/manikandan/CtxBotCodeBase/PyCtxbot_Modified/Weka/model_60logistic.model"));
                ObjectInputStream ois = new ObjectInputStream(
                        new FileInputStream("F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\logisticsmotetwosuernew.model"));
                Logistic cls = (Logistic) ois.readObject();

                ois.close();
                int counter = 1;

                    System.out.println("Starting for:" + username);
                    //Iterate through Query
                    while (cursor.hasNext()) {
                        DBObject doc = cursor.next();
                        BasicDBList basicDBList = (BasicDBList) ((BasicDBObject) doc).get(username);
                        int classcounter = 1;


                        int totalpositives = 0;
                        //Iterate through each query
                        for (Object bquery : basicDBList)
                        {

                            //if (querycounter > 50)
                            //{
                                boolean totalpositveflag = true;
                                try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter( "/home/mvijaya2/ThesisCodeBase/logisticsresults/" + username + ".txt", true))))
                                {
                                    boolean flag = true;

                                    String tweetid = ((BasicDBObject) bquery).get("id").toString();
                                    String tweettext = ((BasicDBObject) bquery).get("text").toString();
                                    Object entitylist = ((BasicDBObject) bquery).get("entities");
                                    Object hashtaglist = ((BasicDBObject) entitylist).get("hashtags");
                                    Object hashtagindex = ((BasicDBList) hashtaglist).get(0);
                                    String qhahstag = ((BasicDBObject) hashtagindex).get("text").toString();
                                    out.println("========================================================================");
                                    out.println(tweetid);
                                    out.println(tweettext);
                                    out.println("#" +qhahstag);
                                    out.println("========================================================================");


                                    DBCollection collindex = indexresultdb.getCollection(tweetid + "_wekaindex");

                                    HashMap<Integer, String> tagindexhashmap = new HashMap<Integer, String>();
                                    HashMap<Integer, String> tweetidindexhashmap = new HashMap<Integer, String>();
                                    DBCursor cursorindex = collindex.find();

                                    while (cursorindex.hasNext()) {
                                        DBObject docobj = cursorindex.next();
                                        Integer index = (Integer) ((BasicDBObject) docobj).get("index");
                                        String hashtag = (String) ((BasicDBObject) docobj).get("tag");
                                        //String tweetText = (String) ((BasicDBObject) docobj).get("tweettext");
                                        String tid = (((BasicDBObject) docobj).get("tweetid")).toString();
                                        tagindexhashmap.put(index, hashtag);
                                        tweetidindexhashmap.put(index, tid);
                                    }

                                    String path = "/home/mvijaya2/ThesisCodeBase/PyCtxbot_Modified/ResultArff/" + username + "/" + tweetid + ".arff";

                                    Instances data = new Instances(new BufferedReader(new FileReader(path)));



                                    //Test Feature Representation
                                    HashMap<Integer,Double> hashtagScoremap = new HashMap<Integer,Double>();
                                    int localindex = 1;
                                    for (int i = 0; i < data.numInstances(); i++)
                                    {
                                        Instance isnt = data.instance(i);
                                        //System.out.println(isnt.toString());
                                        data.setClassIndex(isnt.numAttributes() - 1);
                                        double[] values = cls.distributionForInstance(isnt);
                                        if(isnt.value(9) == 1)
                                        {
                                            if(totalpositveflag){
                                                totalpositives++;
                                                totalpositveflag = false;
                                            }
                                        }
                                        double classy = cls.classifyInstance(isnt);

                                        if (classy == 1)
                                        {
                                            if(hashtagScoremap.containsKey(localindex))
                                            {
                                                double oldscore = hashtagScoremap.get(localindex);
                                                double newscore = values[1];
                                                if (newscore > oldscore){
                                                    hashtagScoremap.put(localindex,newscore);
                                                }
                                            }
                                            else{
                                                hashtagScoremap.put(localindex,values[1]);
                                            }
                                        }
                                        localindex++;
                                    }

                                    HashMap<Integer,Double> sortedscorehashmap = GetsortedhashmapValues(hashtagScoremap);


                                    ArrayList<Integer> kvalues = new ArrayList<>();
                                    kvalues.add(5);
                                    kvalues.add(10);
                                    kvalues.add(15);
                                    kvalues.add(20);
                                    for (int k : kvalues)
                                    {
                                        ListIterator<Map.Entry<Integer, Double>> iter =
                                                new ArrayList(sortedscorehashmap.entrySet()).listIterator(sortedscorehashmap.size());
                                        out.println("**************************************** K = " + k);
                                        int printcounter = 0;
                                        while (iter.hasPrevious())
                                        {
                                            if (printcounter < k)
                                            {
                                                Map.Entry<Integer, Double> entry = iter.previous();
                                                String tweetidxnum = tweetidindexhashmap.get(entry.getKey());
                                                String hahstagresult = tagindexhashmap.get(entry.getKey());

                                                if (hahstagresult == qhahstag){
                                                    out.println(tweetidxnum + ":" + hahstagresult + " " + entry.getKey() + ":" + entry.getValue() + "Rank:" + printcounter + "**********************************************") ;
                                                    if( kresultmap.containsKey(k)){
                                                        int oldvalue =kresultmap.get(k);
                                                        kresultmap.put(k,oldvalue+1);
                                                    }
                                                    else {
                                                        kresultmap.put(k,1);
                                                    }
                                                }
                                                else {
                                                    out.println(tweetidxnum + ":" + hahstagresult + " " + entry.getKey() + ":" + entry.getValue() + "Rank:" + printcounter) ;
                                                }
                                                printcounter++;
                                            }
                                            else{
                                              break;
                                            }
                                        }
                                    }
                                }
                                catch (IOException e) {
                                    //exception handling left as an exercise for the reader
                                    e.printStackTrace();
                                    continue;
                                }

                        }
                        try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter( "/home/mvijaya2/ThesisCodeBase/logisticsresults/" + username + ".txt", true)))) {
                            int five = 0;
                            if (kresultmap.containsKey(5))
                            {
                                five = kresultmap.get(5);
                            }
                            int ten = 0;
                            if (kresultmap.containsKey(5))
                            {
                                ten = kresultmap.get(5);
                            }
                            int fifteen = 0;
                            if (kresultmap.containsKey(5))
                            {
                                fifteen = kresultmap.get(5);
                            }
                            int twenty = 0;
                            if (kresultmap.containsKey(5))
                            {
                                twenty = kresultmap.get(5);
                            }
                            out.println("********************** Final result *******************************");
                            out.println("K=5 " + five + "/" + basicDBList.size() + "/" + totalpositives);
                            out.println("K=10 " + ten + "/" + basicDBList.size() + "/" + totalpositives);
                            out.println("K=15 " + fifteen + "/" + basicDBList.size() + "/" + totalpositives);
                            out.println("K=20 " + twenty + "/" + basicDBList.size() + "/" + totalpositives);
                            out.println(" ");
                        }
                        catch (IOException e) {
                            //exception handling left as an exercise for the reader
                            e.printStackTrace();
                            continue;
                        }

                }

            }
            System.out.println("Done");
        }
        catch (Exception ex) {
            System.out.println(ex);
            ex.printStackTrace();
        }
    }

    public static void StoreLogisticModel(String smotefilepath,String outptusmotemodel) {
        try
        {
            Instances data = new Instances(new BufferedReader(new FileReader(smotefilepath)));
            System.out.println("Started Reading Data");
            data.setClassIndex(data.numAttributes() - 1);

            Logistic model = new Logistic();
            System.out.println("Modeling Data");
            model.buildClassifier(data); //the last instance with missing
       
            // serialize model
            System.out.println("Done Modeling Data");
            ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream(outptusmotemodel));
            oos.writeObject(model);
            oos.flush();
            oos.close();
            //weka.core.SerializationHelper.write("/Users/manikandan/CtxBotCodeBase/PyCtxbot_Modified/Weka/model_60logregression.model", model);
            Instance isnt = data.lastInstance();

            double classy = model.classifyInstance(isnt);
            System.out.println("Score (" + isnt + "): " + classy);

            System.out.println(model);
        }
        catch (Exception ex) {
            System.out.println(ex);
            ex.printStackTrace();
        }

    }

    public static void StoreModelsforDataset()
    {
        try
        {
            //List<Double> percentages = Arrays.asList(100.0,200.0,300.0,400.0,500.0);

                Instances data = new Instances(new BufferedReader(new FileReader("/home/mvijaya2/ThesisCodeBase/smotetraining60dataset.arff")));
                System.out.println("Started Reading Data");
                data.setClassIndex(data.numAttributes() - 1);

                Logistic model = new Logistic();
                System.out.println("Modeling Data");
                model.buildClassifier(data); //the last instance with missing

                // serialize model
                ObjectOutputStream oos = new ObjectOutputStream(
                        new FileOutputStream("/home/mvijaya2/ThesisCodeBase/smotetraningmodel.model"));
                oos.writeObject(model);
                oos.flush();
                oos.close();

                System.out.println("Saving Model");
                //weka.core.SerializationHelper.write("/Users/manikandan/CtxBotCodeBase/PyCtxbot_Modified/Weka/model_60logregression.model", model);
                Instance isnt = data.lastInstance();

                double classy = model.classifyInstance(isnt);
                System.out.println("Model ("+isnt+"): "+classy);

                System.out.println(model);

                System.out.println("Done for modleing");

        }
        catch (Exception ex) {
            System.out.println(ex);
        }

    }

    public static LinkedHashMap GetsortedhashmapValues(HashMap passedMap) {
        List mapKeys = new ArrayList(passedMap.keySet());
        List mapValues = new ArrayList(passedMap.values());
        Collections.sort(mapValues);
        Collections.sort(mapKeys);

        LinkedHashMap sortedMap = new LinkedHashMap();

        Iterator valueIt = mapValues.iterator();
        while (valueIt.hasNext()) {
            Object val = valueIt.next();
            Iterator keyIt = mapKeys.iterator();

            while (keyIt.hasNext()) {
                Object key = keyIt.next();
                String comp1 = passedMap.get(key).toString();
                String comp2 = val.toString();

                if (comp1.equals(comp2)){
                    passedMap.remove(key);
                    mapKeys.remove(key);
                    sortedMap.put((Integer)key, (Double)val);
                    break;
                }

            }

        }
        return sortedMap;
    }

}
