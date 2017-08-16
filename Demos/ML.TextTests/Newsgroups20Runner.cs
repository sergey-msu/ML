using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ML.Core;
using ML.TextMethods.Algorithms;

namespace ML.TextTests
{
  public class Newsgroups20Runner : Runner
  {
    public const string ALT_ATHEISM              = "alt.atheism";
    public const string COMP_GRAPHICS            = "comp.graphics";
    public const string COMP_OS_MS_WINDOWS_MISC  = "comp.os.ms-windows.misc";
    public const string COMP_SYS_IBM_PC_HARDWARE = "comp.sys.ibm.pc.hardware";
    public const string COMP_SYS_MAC_HARDWARE    = "comp.sys.mac.hardware";
    public const string COMP_WINDOWS_X           = "comp.windows.x";
    public const string MISC_FORSALE             = "misc.forsale";
    public const string REC_AUTOS                = "rec.autos";
    public const string REC_MOTORCYCLES          = "rec.motorcycles";
    public const string REC_SPORT_BASEBALL       = "rec.sport.baseball";
    public const string REC_SPORT_HOCKEY         = "rec.sport.hockey";
    public const string SCI_CRYPT                = "sci.crypt";
    public const string SCI_ELECTRONICS          = "sci.electronics";
    public const string SCI_MED                  = "sci.med";
    public const string SCI_SPACE                = "sci.space";
    public const string SOC_RELIGION_CHRISTIAN   = "soc.religion.christian";
    public const string TALK_POLITICS_GUNS       = "talk.politics.guns";
    public const string TALK_POLITICS_MIDEAST    = "talk.politics.mideast";
    public const string TALK_POLITICS_MISC       = "talk.politics.misc";
    public const string TALK_RELIGION_MISC       = "talk.religion.misc";

    public readonly char[] SEPARATOR = new[] { '\t' };

    private Dictionary<string, Class> m_Classes = new Dictionary<string, Class>()
    {
      { ALT_ATHEISM,              new Class("alt.atheism",              0)  },
      { COMP_GRAPHICS,            new Class("comp.graphics",            1)  },
      { COMP_OS_MS_WINDOWS_MISC,  new Class("comp.os.ms-windows.misc",  2)  },
      { COMP_SYS_IBM_PC_HARDWARE, new Class("comp.sys.ibm.pc.hardware", 3)  },
      { COMP_SYS_MAC_HARDWARE,    new Class("comp.sys.mac.hardware",    4)  },
      { COMP_WINDOWS_X,           new Class("comp.windows.x",           5)  },
      { MISC_FORSALE,             new Class("misc.forsale",             6)  },
      { REC_AUTOS,                new Class("rec.autos",                7)  },
      { REC_MOTORCYCLES,          new Class("rec.motorcycles",          8)  },
      { REC_SPORT_BASEBALL,       new Class("rec.sport.baseball",       9)  },
      { REC_SPORT_HOCKEY,         new Class("rec.sport.hockey",        10)  },
      { SCI_CRYPT,                new Class("sci.crypt",               11)  },
      { SCI_ELECTRONICS,          new Class("sci.electronics",         12)  },
      { SCI_MED,                  new Class("sci.med",                 13)  },
      { SCI_SPACE,                new Class("sci.space",               14)  },
      { SOC_RELIGION_CHRISTIAN,   new Class("soc.religion.christian",  15)  },
      { TALK_POLITICS_GUNS,       new Class("talk.politics.guns",      16)  },
      { TALK_POLITICS_MIDEAST,    new Class("talk.politics.mideast",   17)  },
      { TALK_POLITICS_MISC,       new Class("talk.politics.misc",      18)  },
      { TALK_RELIGION_MISC,       new Class("talk.religion.misc",      19)  }
    };

    public override string SrcMark    { get { return "original"; } }
    public override string DataPath   { get { return RootPath+@"\data\newsgroups20"; }}
    public override string OutputPath { get { return RootPath+@"\output\newsgroups20_original"; }}

    protected override TextAlgorithmBase CreateAlgorithm()
    {
      return Examples.Create_Newsgroups20Algorithm();
    }

    #region Export

    protected override void Export()
    {
      throw new NotImplementedException();
    }

    #endregion

    #region Load

    protected override void Load()
    {
      Console.WriteLine("load train data...");
      var trainPath = Path.Combine(SrcPath, "train.txt");
      doLoad(trainPath, m_TrainingSet);

      Console.WriteLine("load test data...");
      var testPath = Path.Combine(SrcPath, "test.txt");
      doLoad(testPath, m_TestingSet);
    }

    private void doLoad(string path, ClassifiedSample<string> sample)
    {
      using (var srcFile = File.Open(path, FileMode.Open, FileAccess.Read))
      using (var srcReader = new StreamReader(srcFile))
      {
        while (true)
        {
          var line = srcReader.ReadLine();
          if (line==null) break;

          var segs = line.Split(SEPARATOR);
          var cls = m_Classes[segs[0].Trim()];
          var doc = segs[1].Trim();

          sample[doc] = cls;
        }
      }
    }

    #endregion
  }
}
