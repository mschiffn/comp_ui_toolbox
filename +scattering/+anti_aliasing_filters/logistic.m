%
% superclass for all logistic spatial anti-aliasing filters
% ( see https://en.wikipedia.org/wiki/Logistic_function )
%
% author: Martin F. Schiffner
% date: 2019-07-31
% modified: 2020-03-04
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
            % 2.) create logistic spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.on( size( growth_rates ) );

            % iterate logistic spatial anti-aliasing filter options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).growth_rate = growth_rates( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = logistic( growth_rates )

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
        function filters = compute_filter( options_anti_aliasing, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.logistic
            if ~isa( options_anti_aliasing, 'scattering.anti_aliasing_filters.logistic' )
                errorStruct.message = 'options_anti_aliasing must be scattering.anti_aliasing_filters.logistic!';
                errorStruct.identifier = 'compute_filter:NoOptionsAntiAliasingLogistic';
                error( errorStruct );
            end

            % ensure cell array for flags
            if ~iscell( flags )
                flags = { flags };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options_anti_aliasing, flags );

            %--------------------------------------------------------------
            % 2.) compute spatial anti-aliasing filters
            %--------------------------------------------------------------
            % specify cell array for filters
            filters = cell( size( options_anti_aliasing ) );

            % iterate spatial anti-aliasing filter options
            for index_filter = 1:numel( options_anti_aliasing )

                % detect valid grid points
                filters{ index_filter } = all( flags{ index_filter } < pi, 3 );

            end % for index_filter = 1:numel( options_anti_aliasing )

            % avoid cell array for single options_anti_aliasing
            if isscalar( options_anti_aliasing )
                filters = filters{ 1 };
            end

        end % function filters = compute_filter( options_anti_aliasing, flags )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( anti_aliasings_logistic )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.logistic
            if ~isa( anti_aliasings_logistic, 'scattering.anti_aliasing_filters.logistic' )
                errorStruct.message = 'anti_aliasings_logistic must be scattering.anti_aliasing_filters.logistic!';
                errorStruct.identifier = 'string:NoOptionsAntiAliasingLogistic';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat empty string for initialization
            strs_out = repmat( "", size( anti_aliasings_logistic ) );

            % iterate raised-cosine spatial anti-aliasing filter options
            for index_object = 1:numel( anti_aliasings_logistic )

                strs_out( index_object ) = sprintf( "logistic (k = %.2f)", anti_aliasings_logistic( index_object ).growth_rate );

            end % for index_object = 1:numel( anti_aliasings_logistic )

        end % function strs_out = string( anti_aliasings_logistic )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute samples of spatial anti-aliasing filter (scalar)
        %------------------------------------------------------------------
        function filter_samples = compute_samples_scalar( filter, flag )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling method ensures class scattering.anti_aliasing_filters.anti_aliasing_filter for filter (scalar)
            % calling method ensures valid flag

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filter (scalar)
            %--------------------------------------------------------------
            % compute logistic function
            filter_samples = prod( 1 ./ ( 1 + exp( filter.growth_rate * ( flag - pi ) ) ), 3 );

        end % function filter_samples = compute_samples_scalar( filter, flag )

	end % methods (Access = protected, Hidden)

end % classdef logistic < scattering.anti_aliasing_filters.on
