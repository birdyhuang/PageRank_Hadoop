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
import java.util.*;

public class PageRank {

    public static final String HDFS_PATH = "hdfs:///user/peng/";
    public static final String OUTPUT_PREFIX = "output";
    public static final String SUMMARY_FILE = "summary.txt";
    public static final int TOP = 10;

    private static float d = 0.85f; // damping factor
    private static float delta = 0.001f; // convergence
    private static int iteration = 0;   // num of iterations
    private static boolean converged = false;   // convergence flag
    private static float initPR = 1.0f; // initial pagerank

    public static HashMap<String, Float> newPR = new HashMap<String, Float>();  // pagerank of previous iteration
    public static HashMap<String, Float> oldPR = newPR; // pagerank of this iteration

    public static HashMap<String, ArrayList<String>> graph = new HashMap<String, ArrayList<String>>();

    private static void println(String str) {
        System.out.println(str);
    }

    /* mapper to of first round */
    public static class PageRankMapperOne extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            ArrayList<String> neighbours = new ArrayList<String>();

            String line = value.toString();

            /* fromID toID1 toID2 toID3 ... toIDn
             * fromID null
             */
            String[] lineSplit = line.split("\\s+");
            String fromID = lineSplit[0];

            long numOutLinks = lineSplit.length - 1;

            StringBuffer sb = new StringBuffer();
            sb.append("|"+initPR+",");

            if (numOutLinks == 0) {
                sb.append("*");
            } else {
                for (int i = 1; i < lineSplit.length; i++) {
                    /* collect each edge
                     * Key: toID        Value: fromID PR numOutLinks
                     */
                    context.write(new Text(lineSplit[i]),
                            new Text(fromID + " " + initPR + " " + numOutLinks));
                    sb.append(lineSplit[i] + " ");
                    neighbours.add(lineSplit[i]);
                }
            }
            /* record the adjacent list */
            if (graph.containsKey(fromID)) {
                graph.get(fromID).addAll(neighbours);
            } else {
                graph.put(fromID, neighbours);
            }

            /* collect all the outLinks from fromID
             * Key: fromID      Value: |PR,toID1 toID2 ...
             * Key: fromID      Value: |PR,*
             */
            context.write(new Text(fromID), new Text(sb.toString()));

        }   // map()
    }   // MapperOne

    /* reducer to construct the adjacent list */
    public static class PageRankReducer extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            float pr = 0.0f;
            String toIDs = null;

            /* Iterate all Text/Text records from the mapper */
            for (Text val : values) {
                String v = val.toString();
                /* Value: |PR,toID1 toID2 ... toIDn
                 * Value: |PR,*
                 */
                if (v.startsWith("|")) {
                    int index = v.indexOf(",");
                    toIDs = v.substring(index+1);  // get outLinks
                    continue;
                }
                /* Value: fromID PR numOutLinks */
                String[] split = v.split("\\s+");
                pr += Float.valueOf(split[1]) / Integer.valueOf(split[2]);
            }

            /* if no other nodes refer to this node,
             *  then this node's pagerank is 0
             */
             pr = 1-d + d*pr;

            /* Key: fromID  Value: PR,toID1 toID2 ... toIDn */
            context.write(key, new Text(pr + "," + toIDs));
            newPR.put(key.toString(), pr);

        }   // reduce()
    }   // ReducerOne

    /* mapper to iteratively compute the pagerank */
    public static class PageRankMapperTwo extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            /* Input
             * Value: fromID    PR,toID1 toID2 ... toIDn
             * Value: fromID    PR,*
             */
            String line = value.toString();

            String[] lineSplit = line.split(",");
            String[] id_pr = lineSplit[0].split("\\s+");

            /* get the fromID */
            String fromID = id_pr[0];

            /* get its pr */
            String pr = id_pr[1];

            /* get its outLinks */
            String[] toIDs = lineSplit[1].split("\\s+");
            if (!toIDs[0].equals("*")) {
                long numOutLinks = toIDs.length;

                for (String toID : toIDs) {
                    /* Output
                     * Key: toID        Value: fromID PR numOutLinks
                     */
                    context.write(new Text(toID), new Text(fromID+" "+pr+" "+numOutLinks));
                }
            }
            /* collect all the toIDs from fromID */
            context.write(new Text(fromID), new Text("|"+pr+","+lineSplit[1]));

        }   // map()
    }   // MapperTwo

    public static boolean runIteration(String inputPath,
                                       String outputPath)
            throws IOException, ClassNotFoundException, InterruptedException {

        /* update the PageRank Map */
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

//        println("Round "+(iteration+1)+" input:"+inputPath);
//        println("Round "+(iteration+1)+" output:"+outputPath);

        if (!job.waitForCompletion(true)) {
            System.err.println("Round "+(iteration+1)+" cannot finish");
            return false;
        }

        System.out.println("Round "+(iteration+1)+" finished");

        return isConverged();
    }

    /* Check the convergence */
    private static boolean isConverged() throws IOException {

        for (Map.Entry<String, Float> entry : newPR.entrySet()) {
            String id = entry.getKey();
            Float f = entry.getValue();
            if (Math.abs(f - oldPR.get(id)) > delta) {
                return false;
            }
        }

        /* differences of all prev & cur prs are ≤ delta */
        return true;
    }

    public static class Node implements Comparable<Node>{
        private String id = null;
        private float pr = 0.0f;

        public Node(String id, float pr) {
            this.id = id;
            this.pr = pr;
        }

        public float getPr() {
            return pr;
        }
        public String getId() {
            return id;
        }
        public void replace(Node newNode) {
            pr = newNode.getPr();
            id = newNode.getId();
        }

        @Override
        public int compareTo(Node node) {
            return (int) ((node.getPr() - pr)*10000);
        }
    }
    public static class Top {
        Node min;
        Node[] nodes;
        int size;

        public Top() {
            min = new Node(null, Float.MAX_VALUE);
            nodes = new Node[TOP];
            size = 0;
        }

        private Node findMin() {
            min = nodes[0];
            for (int i = 1; i < TOP; i ++) {
                if (nodes[i].getPr() < min.getPr()) {
                    min = nodes[i];
                }
            }
            return min;
        }
        public void insert(Node newNode) {
            if (size == TOP) {
                if (newNode.getPr() > min.getPr()) {
                    min.replace(newNode);
                    min = findMin();
                }
            } else {
                nodes[size] = newNode;
                if (min.getPr() > newNode.getPr()) {
                    min = newNode;
                }
                size ++;
            }
        }
    }
    /* Summary info of the graphs */
    public static void summary(long start, long ten, long end) throws IOException {
        long time = end - start;
        long numEdges = 0;
        float avg;
        long min = Long.MAX_VALUE, max = Long.MIN_VALUE;
        String minNode=null, maxNode=null;

        Top top = new Top();

        Formatter f = new Formatter(SUMMARY_FILE);

        f.format("SUMMARY FILE\n\n");
        f.format("%-50s %-20d\n", "Number of nodes:", graph.size());
        f.format("%-50s %-20d\n", "Number of iterations:", iteration);
        f.format("%-50s %-20d\n", "Total execution time(us):", time);
        if (ten == 0) {
            f.format("%-50s\n", "Execution time for the first 10 iterations: less than 10 iterations");
        } else {
            f.format("%-50s %-20d\n", "Execution time for the first 10 iterations(us):", ten - start);
        }
        f.format("%-50s %-20.5f\n", "Average time/iteration:", (float)time/iteration);
        for (Map.Entry<String, ArrayList<String>> entry : graph.entrySet()) {
            ArrayList<String> list = entry.getValue();
            numEdges += list.size();
            if (min > list.size()) {
                min = list.size();
                minNode = entry.getKey();
            }
            if (max < list.size()) {
                max = list.size();
                maxNode = entry.getKey();
            }
            Node node = new Node(entry.getKey(), newPR.get(entry.getKey()));
            top.insert(node);
        }
        avg = (float)numEdges / graph.size();
        f.format("%-50s %-20d\n", "Number of edges:", numEdges);
        f.format("%-50s NodeID: %-45s Out-degree: %-5d\n", "Min out-degree node:", minNode, min);
        f.format("%-50s NodeID: %-45s Out-degree: %-5d\n", "Max out-degree node:", maxNode, max);
        f.format("%-50s %-20.5f\n", "Average out-degree:", avg);

        Arrays.sort(top.nodes);
        f.format("\n");
        f.format("----------------------------TOP 10 PageRank Nodes---------------------------\n");
        for (int i = 0; i < TOP; i ++) {
            f.format("%-50s %-20.5f\n", top.nodes[i].getId(), top.nodes[i].getPr());
        }

        f.flush();
        f.close();
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

        /* start */
        long start = System.currentTimeMillis();

        if (!job.waitForCompletion(true)) {
            System.err.println("Cannot finish the first round job");
        } else {
            System.out.println("Round 1 finished");
        }
        iteration ++;

        job = Job.getInstance();
        job.setMapperClass(PageRankMapperTwo.class);
        job.setReducerClass(PageRankReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        long ten = 0;

        while (!converged) {
            converged = runIteration(args[1] + OUTPUT_PREFIX + iteration,
                                     args[1]+ OUTPUT_PREFIX + (iteration+1));
            iteration ++;
            if (iteration == 10) {
                ten = System.currentTimeMillis();
            }
        }
        System.out.println("Page rank calculation converged!!! All jobs finished");

        /* end */
        long end = System.currentTimeMillis();

        summary(start, ten, end);

        System.exit(0);
    }
}

