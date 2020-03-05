%
% superclass for all raised-cosine spatial anti-aliasing filters
% ( see https://en.wikipedia.org/wiki/Raised-cosine_filter )
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2020-03-04
%
classdef raised_cosine < scattering.anti_aliasing_filters.on

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        roll_off_factor ( 1, 1 ) double { mustBePositive, mustBeLessThanOrEqual( roll_off_factor, 1 ), mustBeNonempty } = 1 % roll-off factor

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = raised_cosine( roll_off_factors )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid roll_off_factors

            %--------------------------------------------------------------
            % 2.) create cosine spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.on( size( roll_off_factors ) );

            % iterate cosine spatial anti-aliasing filter options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).roll_off_factor = roll_off_factors( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = raised_cosine( roll_off_factors )

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
        function filters = compute_filter( options_anti_aliasing, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.raised_cosine
            if ~isa( options_anti_aliasing, 'scattering.anti_aliasing_filters.raised_cosine' )
                errorStruct.message = 'options_anti_aliasing must be scattering.anti_aliasing_filters.raised_cosine!';
                errorStruct.identifier = 'compute_filter:NoOptionsAntiAliasingRaisedCosine';
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

                %----------------------------------------------------------
                % ii.) raised-cosine spatial anti-aliasing filter
                %----------------------------------------------------------
% TODO: small value of options_anti_aliasing( index_filter ).roll_off_factor causes NaN
% TODO: why more conservative aliasing
                % compute lower and upper bounds
                flag_lb = pi * ( 1 - options_anti_aliasing( index_filter ).roll_off_factor );
                flag_ub = pi; %pi * ( 1 + options_anti_aliasing( index_filter ).roll_off_factor );
                flag_delta = flag_ub - flag_lb;

                % detect tapered grid points
                indicator_on = flags{ index_filter } <= flag_lb;
                indicator_taper = ( flags{ index_filter } > flag_lb ) & ( flags{ index_filter } < flag_ub );
                indicator_off = flags{ index_filter } >= flag_ub;

                % compute raised-cosine function
                flags{ index_filter }( indicator_on ) = 1;
                flags{ index_filter }( indicator_taper ) = 0.5 * ( 1 + cos( pi * ( flags{ index_filter }( indicator_taper ) - flag_lb ) / flag_delta ) );
                flags{ index_filter }( indicator_off ) = 0;

                filters{ index_filter } = prod( flags{ index_filter }, 3 );

            end % for index_filter = 1:numel( options_anti_aliasing )

            % avoid cell array for single options_anti_aliasing
            if isscalar( options_anti_aliasing )
                filters = filters{ 1 };
            end

        end % function filters = compute_filter( options_anti_aliasing, flags )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( anti_aliasings_raised_cosine )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.raised_cosine
            if ~isa( anti_aliasings_raised_cosine, 'scattering.anti_aliasing_filters.raised_cosine' )
                errorStruct.message = 'anti_aliasings_raised_cosine must be scattering.anti_aliasing_filters.raised_cosine!';
                errorStruct.identifier = 'string:NoOptionsAntiAliasingRaisedCosine';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "", size( anti_aliasings_raised_cosine ) );

            % iterate raised-cosine spatial anti-aliasing filter options
            for index_object = 1:numel( anti_aliasings_raised_cosine )

                strs_out( index_object ) = sprintf( "raised-cosine (beta = %.2f)", anti_aliasings_raised_cosine( index_object ).roll_off_factor );

            end % for index_object = 1:numel( anti_aliasings_raised_cosine )

        end % function strs_out = string( anti_aliasings_raised_cosine )

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
% TODO: small value of filter.roll_off_factor causes NaN
% TODO: why more conservative aliasing
            % compute lower and upper bounds
            flag_lb = pi * ( 1 - filter.roll_off_factor );
            flag_ub = pi; %pi * ( 1 + filter.roll_off_factor );
            flag_delta = flag_ub - flag_lb;
% TODO: filter function for processing.field?
            % detect tapered grid points
            flag = flag.samples;
            indicator_on = flag <= flag_lb;
            indicator_taper = ( flag > flag_lb ) & ( flag < flag_ub );
            indicator_off = flag >= flag_ub;

            % compute raised-cosine function
            flag( indicator_on ) = 1;
            flag( indicator_taper ) = 0.5 * ( 1 + cos( pi * ( flag( indicator_taper ) - flag_lb ) / flag_delta ) );
            flag( indicator_off ) = 0;
            filter_samples = prod( flag, 3 );

        end % function filter_samples = compute_samples_scalar( filter, flag )

	end % methods (Access = protected, Hidden)

end % classdef raised_cosine < scattering.anti_aliasing_filters.on
