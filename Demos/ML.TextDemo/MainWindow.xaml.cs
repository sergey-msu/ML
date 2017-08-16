using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using NFX;
using NFX.Serialization;
using ML.Core;
using ML.TextMethods.Algorithms;
using ML.TextMethods.Preprocessing;
using ML.TextMethods.Tokenization;
using ML.TextMethods.Stopwords;
using ML.TextMethods.Normalization;
using ML.TextMethods.Stemming;

namespace ML.TextDemo
{
  /// <summary>
  /// Interaction logic for MainWindow.xaml
  /// </summary>
  public partial class MainWindow : Window
  {
    public MainWindow()
    {
      InitializeComponent();

      init();
    }

    private ClassificationAlgorithmBase<string> m_SpamAlgorithm;
    private ClassificationAlgorithmBase<string> m_ReutersR8Algorithm;
    private ClassificationAlgorithmBase<string> m_Newsgroups20Algorithm;

    #region Init

    private void init()
    {
      using (var spam = Assembly.GetExecutingAssembly().GetManifestResourceStream("ML.TextDemo.data.SPAM_p2.42.mld"))
      {
        var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                        new EnglishStopwords(),
                                        new EnglishSimpleNormalizer(),
                                        new EnglishPorterStemmer());
        m_SpamAlgorithm = new ComplementNaiveBayesianAlgorithm(proc);
        m_SpamAlgorithm.Deserialize(spam);
      }

      using (var r8 = Assembly.GetExecutingAssembly().GetManifestResourceStream("ML.TextDemo.data.RR8_p4.37.mld"))
      {
        var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                        new EnglishStopwords(),
                                        new EnglishSimpleNormalizer(),
                                        new EnglishPorterStemmer());
        m_ReutersR8Algorithm = new MultinomialNaiveBayesianAlgorithm(proc);
        m_ReutersR8Algorithm.Deserialize(r8);
      }

      using (var n20 = Assembly.GetExecutingAssembly().GetManifestResourceStream("ML.TextDemo.data.N20_p17.35.mld"))
      {
        var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                        new EnglishStopwords(),
                                        new EnglishSimpleNormalizer(),
                                        new EnglishPorterStemmer());
        m_Newsgroups20Algorithm = new TFIDFNaiveBayesianAlgorithm(proc);
        m_Newsgroups20Algorithm.Deserialize(n20);
      }
    }

    #endregion

    #region Spam

    private void m_SpamClassify_Click(object sender, RoutedEventArgs e)
    {
      var text = m_SpamInput.Text;
      var result = m_SpamAlgorithm.Predict(text);
      m_SpamResult.Text = (result.Value==0) ? "Spam" : "Not spam";
    }

    #endregion

    #region Reuters R8

    private void m_RR8Classify_Click(object sender, RoutedEventArgs e)
    {
      var result = m_ReutersR8Algorithm.Predict(m_RR8Input.Text);
      string text;

      switch (result.Value)
      {
        case 0: text = "Acquisition"; break;
        case 1: text = "Crude"; break;
        case 2: text = "Earnings and Investments"; break;
        case 3: text = "Grain"; break;
        case 4: text = "Interest Rates"; break;
        case 5: text = "Money"; break;
        case 6: text = "Ship"; break;
        case 7: text = "Trade"; break;
        default: text = "...unknown..."; break;
      }

      m_RR8Result.Text = text;
    }

    #endregion

    #region 20 newsgroups

    private void m_N20Classify_Click(object sender, RoutedEventArgs e)
    {
      var result = m_Newsgroups20Algorithm.Predict(m_N20Input.Text);
      string text;

      switch (result.Value)
      {
        case 0:  text = "Religion > Atheism"; break;
        case 1:  text = "Computers > Graphics"; break;
        case 2:  text = "Computers > MS Windows OS"; break;
        case 3:  text = "Computers > IBM Hardware"; break;
        case 4:  text = "Computers > Mac Hardware"; break;
        case 5:  text = "Computers > Windows X"; break;
        case 6:  text = "For Sale"; break;
        case 7:  text = "Autos"; break;
        case 8:  text = "Motorcycles"; break;
        case 9:  text = "Sport > Baseball"; break;
        case 10: text = "Sport > Hockey"; break;
        case 11: text = "Science > Cryptography"; break;
        case 12: text = "Science > Electronics"; break;
        case 13: text = "Science > Medicine"; break;
        case 14: text = "Science > Space"; break;
        case 15: text = "Religion > Christian"; break;
        case 16: text = "Politics > Guns"; break;
        case 17: text = "Politics > Mideast"; break;
        case 18: text = "Politics > Misc"; break;
        case 19: text = "Religion > Misc"; break;
        default: text = "...unknown..."; break;
      }

      m_N20Result.Text = text;
    }

    #endregion

  }
}
