%
% superclass for all logistic spatial anti-aliasing filter options
% ( see https://en.wikipedia.org/wiki/Logistic_function )
%
% author: Martin F. Schiffner
% date: 2019-07-31
% modified: 2019-08-03
%
classdef anti_aliasing_logistic < scattering.options.anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        growth_rate ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 5	% logistic growth rate / steepness of the curve

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing_logistic( growth_rates )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid growth_rates

            %--------------------------------------------------------------
            % 2.) create logistic spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.anti_aliasing( size( growth_rates ) );

            % iterate logistic spatial anti-aliasing filter options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).growth_rate = growth_rates( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = anti_aliasing_logistic( growth_rates )

	end % methods

end % classdef anti_aliasing_logistic < scattering.options.anti_aliasing