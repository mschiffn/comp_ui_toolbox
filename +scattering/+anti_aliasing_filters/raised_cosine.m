%
% superclass for all raised-cosine spatial anti-aliasing filters
% ( see https://en.wikipedia.org/wiki/Raised-cosine_filter )
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2020-03-09
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
            % 2.) create raised-cosine spatial anti-aliasing filters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.on( size( roll_off_factors ) );

            % iterate raised-cosine spatial anti-aliasing filters
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).roll_off_factor = roll_off_factors( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = raised_cosine( roll_off_factors )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( filters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.raised_cosine
            if ~isa( filters, 'scattering.anti_aliasing_filters.raised_cosine' )
                errorStruct.message = 'filters must be scattering.anti_aliasing_filters.raised_cosine!';
                errorStruct.identifier = 'string:NoRaisedCosineSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat empty string
            strs_out = repmat( "", size( filters ) );

            % iterate raised-cosine spatial anti-aliasing filters
            for index_object = 1:numel( filters )

                strs_out( index_object ) = sprintf( "raised-cosine (beta = %.2f)", filters( index_object ).roll_off_factor );

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
        function samples = compute_samples_scalar( filter, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling method ensures class scattering.anti_aliasing_filters.anti_aliasing_filter for filter (scalar)
            % calling method ensures valid flags

            %--------------------------------------------------------------
            % 2.) compute filter samples (scalar)
            %--------------------------------------------------------------
% TODO: small value of filter.roll_off_factor causes NaN
% TODO: why more conservative aliasing
            % compute lower and upper bounds
            flag_lb = pi * ( 1 - filter.roll_off_factor );
            flag_ub = pi; %pi * ( 1 + filter.roll_off_factor );
            flag_delta = flag_ub - flag_lb;

            % detect tapered grid points
            indicator_on = flags <= flag_lb;
            indicator_taper = ( flags > flag_lb ) & ( flags < flag_ub );
            indicator_off = flags >= flag_ub;

            % compute raised-cosine function
            flags( indicator_on ) = 1;
            flags( indicator_taper ) = 0.5 * ( 1 + cos( pi * ( flags( indicator_taper ) - flag_lb ) / flag_delta ) );
            flags( indicator_off ) = 0;
            samples = prod( flags, 3 );

        end % function samples = compute_samples_scalar( filter, flags )

	end % methods (Access = protected, Hidden)

end % classdef raised_cosine < scattering.anti_aliasing_filters.on
