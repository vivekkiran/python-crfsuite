#ifndef TAGGER_WRAPPER_H
#define TAGGER_WRAPPER_H 1

#include <stdio.h>
#include <errno.h>
#include <stdexcept>
#include "crfsuite_api.hpp"


namespace CRFSuiteWrapper
{

/**
* A wrapper around CRFSuite::Tagger that allows to call 'dump' method
* from Python.
*/
class Tagger : public CRFSuite::Tagger
{
public:
    /**
    * CRFSuite::Tagger::viterbi with a guard for models without labels
    * (e.g. trained without any data). CRFsuite's Viterbi decoder assumes
    * at least one label and reads out of bounds otherwise, see
    * https://github.com/scrapinghub/python-crfsuite/issues/28
    */
    CRFSuite::StringList viterbi()
    {
        if (model != NULL && tagger != NULL && tagger->length(tagger) > 0) {
            crfsuite_dictionary_t *labels = NULL;
            if (model->get_labels(model, &labels)) {
                throw std::runtime_error("Failed to obtain the dictionary interface for labels");
            }
            int num_labels = labels->num(labels);
            labels->release(labels);
            if (num_labels <= 0) {
                throw std::invalid_argument(
                    "The model has no labels (was it trained without any data?)");
            }
        }
        return CRFSuite::Tagger::viterbi();
    }

    // The inherited tag() would call the unguarded base-class viterbi().
    CRFSuite::StringList tag(const CRFSuite::ItemSequence& xseq)
    {
        set(xseq);
        return viterbi();
    }

    void dump(int fileno)
    {
        if (model == NULL) {
            throw std::runtime_error("Tagger is closed");
        }

        FILE* file = fdopen(fileno, "w");
        if (!file){
            throw std::runtime_error("Can't open file");
        }

        model->dump(model, file);

        if (fclose(file)){
            throw std::runtime_error("Can't close file");
        };
    }
};

}
#endif
