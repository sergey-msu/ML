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

    #region Init

    private void init()
    {
      using (var spam = Assembly.GetExecutingAssembly().GetManifestResourceStream("ML.TextDemo.data.SPAM_p2,52.mld"))
      {
        var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                        new EnglishStopwords(),
                                        new EnglishSimpleNormalizer(),
                                        new EnglishPorterStemmer());
        m_SpamAlgorithm = new ComplementNaiveBayesianAlgorithm(proc);
        m_SpamAlgorithm.Deserialize(spam);
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

  }
}
