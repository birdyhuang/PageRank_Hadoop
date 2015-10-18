package org.cloudcomputing.pagerank;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class PageRank {

    public static final String HDFS_PATH = "hdfs:///user/peng/";
    public static final String OUTPUT_PREFIX = "output";

    private static float d = 0.85f; // damping factor
    private static float delta = 0.001f; // convergence
    private static int iteration = 0;   // num of iterations
    private static boolean converged = false;   // convergence flag
    private static float initPR = 1.0f; // initial pagerank

    public static HashMap<String, Float> newPR = new HashMap<String, Float>();  // pagerank of previous iteration
    public static HashMap<String, Float> oldPR = newPR; // pagerank of this iteration

    public static void println(String a) {
        System.out.println(a);
    }
    public static void print(HashMap<String, Float> map) {
        for (Map.Entry<String, Float> entry : map.entrySet()) {
            println("map:"+entry.getKey()+":"+entry.getValue());
        }
    }

    // mapper to of first round
    public static class PageRankMapperOne extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();

            // fromID toID1 toID2 toID3 ... toIDn
            // fromID null
            String[] lineSplit = line.split("\\s+");
            String fromID = lineSplit[0];

            long numOutLinks = lineSplit.length - 1;

            StringBuffer sb = new StringBuffer();
            sb.append("|"+initPR+",");

            if (numOutLinks == 0) {
                sb.append("*");
            } else {
//              int index = line.indexOf(" ");
//              String toIDs = line.substring(index+1);
                for (int i = 1; i < lineSplit.length; i++) {
                    // Collect each edge
                    // Key: toID    Value: fromID PR numOutLinks
                    context.write(new Text(lineSplit[i]), new Text(fromID + " " + initPR + " " + numOutLinks));
                    sb.append(lineSplit[i] + " ");
                }
            }

            // collect all the outLinks from fromID
            // Key: fromID      Value: |PR,toID1 toID2 ...
            // Key: fromID      Value: |PR,*
            context.write(new Text(fromID), new Text(sb.toString()));
//            context.write(new Text(fromID), new Text("|"+toIDs));

        }   // map()
    }   // MapperOne

    // reducer to construct the adjacent list
    public static class PageRankReducer extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            float pr = 0.0f;
            String toIDs = null;

            // Iterate all Text/Text records from the mapper
//            while (value.hasNext()) {
//                String next = value.next().toString();
//                // Value: |PR,toID1 toID2 ... toIDn
//                if (next.startsWith("|")) {
//                    sb.append(next.substring(1));
//                    continue;
//                }
//                // Value: fromID PR outLinks
//                String[] split = next.split("\\s+");
//                pr += Float.valueOf(split[1]) / Integer.valueOf(split[2]);
//            }
            for (Text val : values) {
                String v = val.toString();
                // Value: |PR,toID1 toID2 ... toIDn
                // Value: |PR,*
                if (v.startsWith("|")) {
                    int index = v.indexOf(",");
//                    pr += Float.valueOf(v.substring(1, index));
                    toIDs = v.substring(index+1);  // get outLinks
                    continue;
                }
                // Value: fromID PR numOutLinks
                String[] split = v.split("\\s+");
                pr += Float.valueOf(split[1]) / Integer.valueOf(split[2]);
            }

            if (pr != 0.0f) {
                pr = 1-d + d*pr;
            }

            // Key: fromID  Value: PR,toID1 toID2 ... toIDn
            context.write(key, new Text(pr + "," + toIDs));
            newPR.put(key.toString(), pr);

        }   // reduce()
    }   // ReducerOne

    // mapper to iteratively compute the pagerank
    public static class PageRankMapperTwo extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            // Input
            // Value: fromID    PR,toID1 toID2 ... toIDn
            // Value: fromID    PR,*
            String line = value.toString();

            String[] lineSplit = line.split(",");
            String[] id_pr = lineSplit[0].split("\\s+");

            // get the fromID
//            String fromID = line.substring(0, index);
            String fromID = id_pr[0];

//            String[] lineSplit = line.substring(index+1).split(",");

            // get its pr
//            String pr = lineSplit[0];
            String pr = id_pr[1];

            // get its outLinks
            String[] toIDs = lineSplit[1].split("\\s+");
            if (!toIDs[0].equals("*")) {
                long numOutLinks = toIDs.length;

                for (String toID : toIDs) {
                    // Output
                    // Key: toID        Value: fromID PR numOutLinks
                    context.write(new Text(toID), new Text(fromID+" "+pr+" "+numOutLinks));
                }
                // collect all the toIDs from fromID
            }
            context.write(new Text(fromID), new Text("|"+pr+","+lineSplit[1]));

        }   // map()
    }   // MapperTwo

    public static boolean runIteration(String inputPath,
                                       String outputPath)
            throws IOException, ClassNotFoundException, InterruptedException {

        oldPR = newPR;
        newPR = new HashMap<String, Float>();

        Job job = Job.getInstance();

        job.setMapperClass(PageRankMapperTwo.class);
        job.setReducerClass(PageRankReducer.class);
        job.setJarByClass(PageRank.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        println("Round "+(iteration+1)+" input:"+inputPath);
        println("Round "+(iteration+1)+" output:"+outputPath);

        if (!job.waitForCompletion(true)) {
            System.err.println("Round "+(iteration+1)+" cannot finish");
            return false;
        }

        System.out.println("Round "+(iteration+1)+" finished");

        return isConverged(inputPath, outputPath);
    }

    private static boolean isConverged(String prev, String cur) throws IOException {

        float max = 0.0f;

//        FileReader frPrev = new FileReader(HDFS_PATH + prev);
//        FileReader frCur = new FileReader(HDFS_PATH + cur);
////        FileReader frCur = new FileReader(HDFS_PATH_OUTPUT+cur.getName());
//
//        BufferedReader brp = new BufferedReader(frPrev);
//        BufferedReader brc = new BufferedReader(frCur);
//
////        FileSystem fs = FileSystem.get(URI.create(prev.toString()), new Configuration());
////        BufferedReader brp = new BufferedReader(new InputStreamReader(fs.open(prev)));
////        BufferedReader brc = new BufferedReader(new InputStreamReader(fs.open(cur)));
//
//        String linep, linec;
//
//
//        while ((linep = brp.readLine()) != null && (linec = brc.readLine()) != null) {
//            max = Math.max(max, compareByLine(linep, linec));
//        }

        for (Map.Entry<String, Float> entry : newPR.entrySet()) {
            String id = entry.getKey();
            Float f = entry.getValue();
//            max = Math.max(max, Math.abs(f - oldPR.get(id)));
//            println(id+":"+f);
            if (Math.abs(f - oldPR.get(id)) > delta) {
                return false;
            }
        }

//        return max <= delta;
        return true;
    }

    private static float compareByLine(String a, String b) {

        // fromID PR,toID1 toID2 ... toIDn
        String[] aSplit = a.split(",");
        float pra = Float.valueOf(aSplit[0].split("\\s+")[1]);

        String[] bSplit = b.split(",");
        float prb = Float.valueOf(bSplit[0].split("\\s+")[1]);

        return Math.abs(pra - prb);
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        if (args.length != 2) {
            System.out.println("Usage: hadoop jar pageRank.jar <inputDir> <outputPath>");
            System.exit(0);
        }
        Job job = Job.getInstance();

//        // InputFormat/OutputFormat: default to be TextInputFormat inherited from FileInputFormat
//        job.setInputFormatClass(FileInputFormat.class);
//        job.setOutputFormatClass();

        job.setMapperClass(PageRankMapperOne.class);
        job.setReducerClass(PageRankReducer.class);
        job.setJarByClass(PageRank.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1] + OUTPUT_PREFIX+ 1));

        if (!job.waitForCompletion(true)) {
            System.err.println("Cannot finish the first round job");
        } else {
            System.out.println("First round job finished");
        }
        iteration ++;

        job = Job.getInstance();
        job.setMapperClass(PageRankMapperTwo.class);
        job.setReducerClass(PageRankReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        while (!converged) {
            converged = runIteration(args[1] + OUTPUT_PREFIX + iteration,
                                     args[1]+ OUTPUT_PREFIX + (iteration+1));
//            converged = runIteration(new Path(args[1]+iteration),
//                                     new Path(args[1]+(iteration+1)));
            iteration ++;
        }
        System.out.println("Page rank calculation converged!!! All jobs finished");
        System.exit(0);
    }
}

