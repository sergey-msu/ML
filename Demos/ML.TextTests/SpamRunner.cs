using System;
using System.Collections.Generic;
using System.IO;
using ML.Core;
using ML.TextMethods.Algorithms;
using System.Text;

namespace ML.TextTests
{
  public class SpamRunner : Runner
  {
    public const string    SPAM = "spam";
    public const string    HAM  = "ham";
    public readonly char[] SEPARATOR = new[] { ',' };


    private Dictionary<string, Class> m_Classes = new Dictionary<string, Class>()
    {
      { SPAM, new Class("Spam",  0) },
      { HAM,  new Class("Ham",   1) }
    };

    public override string SrcMark    { get { return "original"; } }
    public override string DataPath   { get { return RootPath+@"\data\spam"; }}
    public override string OutputPath { get { return RootPath+@"\output\spam_original"; }}

    protected override TextAlgorithmBase CreateAlgorithm()
    {
      return Examples.Create_TWCAlgorithm();
    }

    #region Export

    protected override void Export()
    {
      var srcPath = Path.Combine(SrcPath, "spam.csv");
      var outPath = Path.Combine(SrcPath, string.Format("spam_{0}.csv", Alg.Name));

      doExport(srcPath, outPath);
    }

    private void doExport(string fpath, string opath)
    {
      var sample = new ClassifiedSample<string>();

      using (var srcFile = File.Open(fpath, FileMode.Open, FileAccess.Read))
      using (var srcReader = new StreamReader(srcFile))
      {
        var line = srcReader.ReadLine();
        var segs = line.Split(SEPARATOR, StringSplitOptions.RemoveEmptyEntries);
        var cls = m_Classes[segs[0]];
        var doc = segs[1];

        sample.Add(doc, cls);
      }

      var vocabulary = Alg.ExtractVocabulary(sample);
      var dim = vocabulary.Count;
      var builder = new StringBuilder();

      using (var outFile = File.Open(opath, FileMode.CreateNew, FileAccess.Write))
      using (var outWriter = new StreamWriter(outFile))
      {
        for (int i=0; i<dim; i++)
          builder.AppendFormat("{0},", vocabulary[i]);
        builder.Append("_class,_value,_training");

        outWriter.WriteLine(builder.ToString());

        foreach (var pData in sample)
        {
          var doc  = pData.Key;
          var cls  = pData.Value;
          var data = Alg.ExtractFeatureVector(doc);

          builder.Clear();
          for (int i=0; i<dim; i++)
            builder.AppendFormat("{0},", data[i]);
          builder.AppendFormat("{0},{1},{2}", cls.Name, cls.Value, 1);

          outWriter.WriteLine(builder.ToString());
        }
      }
    }

    #endregion

    #region Load

    protected override void Load()
    {
      Console.WriteLine("load train data...");
      var srcPath = Path.Combine(SrcPath, "spam.csv");

      doLoad(srcPath);
    }

    private void doLoad(string path)
    {
      var sample = new ClassifiedSample<string>();

      using (var srcFile = File.Open(path, FileMode.Open, FileAccess.Read))
      using (var srcReader = new StreamReader(srcFile))
      {
        while (true)
        {
          var line = srcReader.ReadLine();
          if (line==null) break;

          line = line.Replace('"', ' ').TrimEnd(SEPARATOR);
          var sIdx = line.IndexOf(SEPARATOR[0]);
          if (sIdx<0) continue;

          var cls = m_Classes[line.Substring(0, sIdx).Trim()];
          var doc = line.Substring(sIdx+1, line.Length-sIdx-1).Trim();

          sample[doc] = cls;
        }
      }

      var cnt = sample.Count;
      var tcnt = cnt*4/5;

      m_TrainingSet = sample.Subset(0, tcnt);
      m_TestingSet  = sample.Subset(tcnt, cnt-tcnt);
    }

    #endregion

    #region Train

    protected override void Train()
    {
      var now = DateTime.Now;

      Console.WriteLine();
      Console.WriteLine("Training started at {0}", now);
      Alg.Train(m_TrainingSet);

      Utils.HandleTrainEnded(Alg, m_TestingSet, OutputPath);

      Console.WriteLine("\n--------- ELAPSED TRAIN ----------" + (int)(DateTime.Now-now).TotalSeconds + "s");
    }

    #endregion

    #region Test

    protected override void Test()
    {
      throw new NotSupportedException();
    }

    #endregion
  }
}
