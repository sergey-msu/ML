using System;
using System.Collections.Generic;

namespace ML.TextTests
{
  class Program
  {
    static void Main(string[] args)
    {
      var runner =
        new SpamRunner();

      try
      {
        Console.WriteLine("STARTED at {0}", DateTime.Now);
        runner.Run();
        Console.WriteLine("DONE at {0}", DateTime.Now);
      }
      catch (Exception ex)
      {
        Console.WriteLine("ERROR:");
        Console.WriteLine(ex);

        Utils.SaveAlgCrushResults(runner.Alg, runner.OutputPath);
      }

      Console.ReadLine();
    }
  }
}
