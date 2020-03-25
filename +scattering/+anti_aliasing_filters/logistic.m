%
% superclass for all logistic spatial anti-aliasing filters
% ( see https://en.wikipedia.org/wiki/Logistic_function )
%
% author: Martin F. Schiffner
% date: 2019-07-31
% modified: 2020-03-09
%
classdef logistic < scattering.anti_aliasing_filters.on

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
        function objects = logistic( growth_rates )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid growth_rates

            %--------------------------------------------------------------
            % 2.) create logistic spatial anti-aliasing filters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.on( size( growth_rates ) );

            % iterate logistic spatial anti-aliasing filters
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).growth_rate = growth_rates( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = logistic( growth_rates )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( filters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.logistic
            if ~isa( filters, 'scattering.anti_aliasing_filters.logistic' )
                errorStruct.message = 'filters must be scattering.anti_aliasing_filters.logistic!';
                errorStruct.identifier = 'string:NoLogisticSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat empty string
            strs_out = repmat( "", size( filters ) );

            % iterate logistic spatial anti-aliasing filters
            for index_object = 1:numel( filters )

                strs_out( index_object ) = sprintf( "logistic (k = %.2f)", filters( index_object ).growth_rate );

            end % for index_object = 1:numel( filters )

        end % function strs_out = string( filters )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute filter samples (scalar)
        %------------------------------------------------------------------
        function samples = compute_samples_scalar( filter, flags_samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling method ensures class scattering.anti_aliasing_filters.anti_aliasing_filter for filter (scalar)
            % calling method ensures valid flags_samples

            %--------------------------------------------------------------
            % 2.) compute filter samples (scalar)
            %--------------------------------------------------------------
            % compute logistic function
            samples = prod( 1 ./ ( 1 + exp( filter.growth_rate * ( flags_samples - pi ) ) ), 3 );

        end % function samples = compute_samples_scalar( filter, flags_samples )

	end % methods (Access = protected, Hidden)

end % classdef logistic < scattering.anti_aliasing_filters.on
