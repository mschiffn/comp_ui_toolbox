%
% superclass for all raised-cosine spatial anti-aliasing filter options
% ( see https://en.wikipedia.org/wiki/Raised-cosine_filter )
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2020-02-01
%
classdef anti_aliasing_raised_cosine < scattering.options.anti_aliasing

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
        function objects = anti_aliasing_raised_cosine( roll_off_factors )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid roll_off_factors

            %--------------------------------------------------------------
            % 2.) create cosine spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.anti_aliasing( size( roll_off_factors ) );

            % iterate cosine spatial anti-aliasing filter options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).roll_off_factor = roll_off_factors( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = anti_aliasing_raised_cosine( roll_off_factors )

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
        function filters = compute_filter( options_anti_aliasing, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.anti_aliasing_raised_cosine
            if ~isa( options_anti_aliasing, 'scattering.options.anti_aliasing_raised_cosine' )
                errorStruct.message = 'options_anti_aliasing must be scattering.options.anti_aliasing_raised_cosine!';
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
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( anti_aliasings_raised_cosine )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.anti_aliasing_raised_cosine
            if ~isa( anti_aliasings_raised_cosine, 'scattering.options.anti_aliasing_raised_cosine' )
                errorStruct.message = 'anti_aliasings_raised_cosine must be scattering.options.anti_aliasing_raised_cosine!';
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

end % classdef anti_aliasing_raised_cosine < scattering.options.anti_aliasing
