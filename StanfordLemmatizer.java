import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class StanfordLemmatizer {

    protected StanfordCoreNLP pipeline;

    public StanfordLemmatizer() {
        // Create StanfordCoreNLP object properties, with POS tagging
        // (required for lemmatization), and lemmatization
        Properties props;
        props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");
      
        this.pipeline = new StanfordCoreNLP(props);
    }

    public List<String> lemmatize(String documentText)
    {
        List<String> lemmas = new LinkedList<String>();
        // Create an empty Annotation just with the given text
        Annotation document = new Annotation(documentText);
        // run all Annotators on this text
        this.pipeline.annotate(document);
        // Iterate over all of the sentences found
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        for(CoreMap sentence: sentences) {
            // Iterate over all tokens in a sentence
            for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                // Retrieve and add the lemma for each word into the
                // list of lemmas
                lemmas.add(token.get(LemmaAnnotation.class));
            }
        }
        return lemmas;
    }


    public static void main(String[] args) {
        System.out.println("Starting Stanford Lemmatizer");

        StanfordLemmatizer slem = new StanfordLemmatizer();
        
        String doc1 = "information retrieval is the most awesome class I ever took";
        String doc2 = "the retrieval of private information from your emails is a job that the NSA loves";
        String doc3 = "in the school of information you can learn about data science";
        String doc4 = "the labrador retriever is a great dog";
        		
        List<String> lemDoc1 = slem.lemmatize(doc1);
        List<String> lemDoc2 = slem.lemmatize(doc2);
        List<String> lemDoc3 = slem.lemmatize(doc3);
        List<String> lemDoc4 = slem.lemmatize(doc4);
        		
        //count number of lemmas
        int collectionCount = lemDoc1.size() + lemDoc2.size() + lemDoc3.size() +lemDoc4.size();

//		System.out.println("Enter query :");
//		String querystr = System.console().readLine();
        
        String querystr = "information retrieval";
        
		List<String> lemQueryStr = slem.lemmatize(querystr);
		
		
		//find p(t|collection) for all terms and store it
		float[] ptc = new float[lemQueryStr.size()];		
		int i=0;
		for(String qlemma: lemQueryStr)
		{
			float freq = (float) (Collections.frequency(lemDoc1, qlemma) + Collections.frequency(lemDoc2, qlemma) + Collections.frequency(lemDoc3, qlemma) + Collections.frequency(lemDoc4, qlemma));
			ptc[i] =  freq / collectionCount;
			i++;
		}
		
		//Find P(q|d) for all docs 
		Double[][] pqd = new Double[4][2];		

		pqd[0][0] = 1.0d;
		pqd[1][0] = 2.0d;
		pqd[2][0] = 3.0d;
		pqd[3][0] = 4.0d;
		
		pqd[0][1] = 1.0d;
		pqd[1][1] = 1.0d;
		pqd[2][1] = 1.0d;
		pqd[3][1] = 1.0d;

		final Comparator<Double[]> arrayComparator = new Comparator<Double[]>() {
	        @Override
	        public int compare(Double[] o1, Double[] o2) {
	            return o2[1].compareTo(o1[1]);
	        }
	    };
		
		// for every lemma in query
	    int lmct =0;
	    for(String qlemma: lemQueryStr)
		{
			
			double lemCountInDoc = (double) Collections.frequency(lemDoc1, qlemma);
			pqd[0][1] *= ((0.5 * (lemCountInDoc/(double)lemDoc1.size())) + (0.5 * ptc[lmct]));			
				
			

			lemCountInDoc = (double) Collections.frequency(lemDoc2, qlemma);
			pqd[1][1] *= ((0.5 * (lemCountInDoc/(double)lemDoc2.size())) + (0.5 * ptc[lmct]));	
			
			
			lemCountInDoc = (double) Collections.frequency(lemDoc3, qlemma);						
			pqd[2][1] *= ((0.5 * (lemCountInDoc/(double)lemDoc3.size())) + (0.5 * ptc[lmct]));
						

			lemCountInDoc = (double) Collections.frequency(lemDoc4, qlemma);			
			pqd[3][1] *= ((0.5 * (lemCountInDoc/(double)lemDoc4.size())) + (0.5 * ptc[lmct]));
			
			lmct ++;
		}
		
		//rank docs according to P(q|d) value
		Arrays.sort(pqd,arrayComparator);
		
		System.out.println("Ranking:");
        System.out.println(pqd[0][0] + " - " + pqd[0][1]);
        System.out.println(pqd[1][0] + " - " + pqd[1][1]);
        System.out.println(pqd[2][0] + " - " + pqd[2][1]);
        System.out.println(pqd[3][0] + " - " + pqd[3][1]);
    }

}