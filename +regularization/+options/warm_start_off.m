%
% superclass for all inactive warm start options
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2020-01-07
%
classdef warm_start_off < regularization.options.warm_start

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = warm_start_off( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create inactive warm start options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.warm_start( size );

        end % function objects = warm_start_off( size )

        %------------------------------------------------------------------
        % display warm start options
        %------------------------------------------------------------------
        function str_out = show( warm_starts_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.warm_start_off
            if ~isa( warm_starts_off, 'regularization.options.warm_start_off' )
                errorStruct.message = 'warm_starts_off must be regularization.options.warm_start_off!';
                errorStruct.identifier = 'show:NoOptionsWarmStartOff';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display options
            %--------------------------------------------------------------
            % specify cell array for str_out
            str_out = repmat( { 'off' }, size( warm_starts_off ) );

            % avoid cell array for single warm_starts_off
            if isscalar( warm_starts_off )
                str_out = str_out{ 1 };
            end

        end % function str_out = show( warm_starts_off )

	end % methods

end % classdef warm_start_off < regularization.options.warm_start
