#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "optimizers/optimizer.h"


class optimizer_factory
{
public:
    using factory_fn = std::function<std::unique_ptr<optimizer>()>;
    using hyperparam_list = std::vector<std::pair<std::string, std::string>>;

    struct result
    {
        factory_fn factory;
        hyperparam_list hyperparams;
    };

    static result create(
        std::string_view algorithm,
        const ihcp_model& model,
        const ihcp_bounds& bounds,
        std::span<char*> optimizer_args
    );
};