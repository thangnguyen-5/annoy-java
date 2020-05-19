package com.spotify.annoy;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

@RunWith(JUnit4.class)
public class ANNIndexTest {

  private static final String DIR = "src/test/resources";
  private static final List<Integer> DIMS = Arrays.asList(64);

  private void testIndex(IndexType type, int blockSize, boolean verbose)
          throws IOException {

    String ts = type.toString().toLowerCase();
    for (int dim : DIMS) {
      System.out.println(String.format("Testing with dim = %d", dim));

      ANNIndex index = new ANNIndex(dim,
              String.format("%s/points.%s.annoy.%d", DIR, ts, dim), type, blockSize);
      BufferedReader reader = new BufferedReader(new FileReader(
              String.format("%s/points.%s.ann.%d.txt", DIR, ts, dim)));
      int acceptable = 0;
      int total = 0;
      while (true) {
        // read in expected results from file (precomputed from c++ version)
        String line = reader.readLine();
        if (line == null)
          break;
        total += 1;
        String[] _l = line.split("\t");
        Integer queryItemIndex = Integer.parseInt(_l[0]);
        List<Integer> expectedResults = new LinkedList<>();
        for (String _i : _l[1].split(","))
          expectedResults.add(Integer.parseInt(_i));

        // do the query
        float[] itemVector = index.getItemVector(queryItemIndex);
        List<Integer> retrievedResults = index.getNearest(itemVector, 21);

        if (verbose) {
          System.out.println(String.format("query: %d", queryItemIndex));
          for (int i = 0; i < 21; i++)
            System.out.println(String.format("expected %6d retrieved %6d",
                    expectedResults.get(i),
                    retrievedResults.get(i)));
          System.out.println();
        }

        Set<Integer> totRes = new TreeSet<>();
        totRes.addAll(expectedResults);
        totRes.retainAll(retrievedResults);
        if (totRes.size() >= 15) {
          acceptable += 1;
        }
      }
      System.out.println(String.format("Acceptable: %s\nTotal: %s\nCoverage: %s", acceptable, total, (double)acceptable / total * 100.0));
      assert (acceptable > total * 95 / 100);
    }
  }

  @Test
  /**
   Make sure that the NNs retrieved by the Java version match the
   ones pre-computed by the C++ version of the Angular index
   using the default block size (for files up to 2GB).
   */
  public void testAngular() throws IOException {
    testIndex(IndexType.ANGULAR, 0, false);
  }

  @Test
  /**
   Make sure that the NNs retrieved by the Java version match the
   ones pre-computed by the C++ version of the Angular index
   using the default block size (for files up to 2GB).
   */
  public void testDot() throws IOException {
    testIndex(IndexType.DOT, 0, false);
  }


  @Test
  /**
   Make sure that the NNs retrieved by the Java version match the
   ones pre-computed by the C++ version of the Euclidean index
   using the default block size (for files up to 2GB).
   */
  public void testEuclidean() throws IOException {
    testIndex(IndexType.EUCLIDEAN, 0, false);
  }

  @Test
  /**
   Make sure that the NNs retrieved by the Java version match the
   ones pre-computed by the C++ version of the Angular index
   simulating files larger than 2GB.
   */
  public void testAngularBlocks() throws IOException {
    testIndex(IndexType.ANGULAR, 10, false);
    testIndex(IndexType.ANGULAR, 1, false);
  }

  @Test
  /**
   Make sure that the NNs retrieved by the Java version match the
   ones pre-computed by the C++ version of the Angular index
   simulating files larger than 2GB.
   */
  public void testDotBlocks() throws IOException {
    testIndex(IndexType.DOT, 10, false);
    testIndex(IndexType.DOT, 1, false);
  }


  @Test
  /**
   Make sure that the NNs retrieved by the Java version match the
   ones pre-computed by the C++ version of the Euclidean index
   simulating files larger than 2GB.
   */
  public void testEuclideanMultipleBlocks() throws IOException {
    testIndex(IndexType.EUCLIDEAN, 10, false);
    testIndex(IndexType.EUCLIDEAN, 1, false);
  }

  @Test(expected = RuntimeException.class)
  /**
   Make sure wrong dimension size used to init ANNIndex will throw RuntimeException.
   */
  public void testLoadFileWithWrongDimension() throws IOException {
    ANNIndex index = new ANNIndex(7, "src/test/resources/points.euclidean.annoy");
  }

  @Test(expected = RuntimeException.class)
  /**
   Make sure wrong dimension size throw exception in getNearest()
   */
  public void testGetNearesWithWrongDim() throws IOException {
    ANNIndex index = new ANNIndex(8, "src/test/resources/points.angular.annoy", IndexType.ANGULAR);
    float[] u = {0f, 1.0f, 0.2f, 0.1f, 0f, 1.0f, 0.2f, 0.1f, 1f};
    index.getNearest(u, 10);
  }
}
