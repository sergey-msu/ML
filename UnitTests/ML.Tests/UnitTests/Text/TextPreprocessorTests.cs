using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.TextMethods;
using ML.TextMethods.Preprocessing;
using ML.TextMethods.Normalization;
using ML.TextMethods.Stemming;
using ML.TextMethods.Stopwords;
using ML.TextMethods.Tokenization;

namespace ML.Tests.UnitTests.Text
{
  [TestClass]
  public class TextPreprocessorTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    [TestMethod]
    public void EnglishTextPreprocessor_Preprocess()
    {
      // arrange
      var preprocessor = new TextPreprocessor(
                               new EnglishSimpleTokenizer(),
                               new EnglishStopwords(),
                               new EnglishSimpleNormalizer(),
                               new EnglishPorterStemmer());
      var text =
      @"Jack London was born on January 12, 1876.  By age 30 London was internationally famous for his books
        Call of the Wild (1903), The Sea Wolf (1904) and other literary and journalistic accomplishments.";

      // act
      var dict = preprocessor.Preprocess(text);

      // assert
      Assert.AreEqual(dict.Count, 15);
      Assert.AreEqual(dict[0],  "jack");
      Assert.AreEqual(dict[1],  "london");
      Assert.AreEqual(dict[2],  "born");
      Assert.AreEqual(dict[3],  "januari");
      Assert.AreEqual(dict[4],  "ag");
      Assert.AreEqual(dict[5],  "london");
      Assert.AreEqual(dict[6],  "internation");
      Assert.AreEqual(dict[7],  "famou");
      Assert.AreEqual(dict[8],  "book");
      Assert.AreEqual(dict[9],  "wild");
      Assert.AreEqual(dict[10], "sea");
      Assert.AreEqual(dict[11], "wolf");
      Assert.AreEqual(dict[12], "literari");
      Assert.AreEqual(dict[13], "journalist");
      Assert.AreEqual(dict[14], "accomplish");
    }

  }
}
